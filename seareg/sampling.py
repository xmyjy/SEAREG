"""Legacy-compatible RNG streams with local peak IDs and bounded mask allocation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np
import torch
from .graph import bipartite_graph
from .config import sampling_config


_MAX_NEGATIVE_SAMPLING_WORKERS = 8


@dataclass
class ShardBatch:
    users: torch.Tensor
    positive: torch.Tensor
    negative: torch.Tensor
    adjacency: torch.Tensor

    def to(self, device, stable=None):
        """Move a shard, optionally reusing device tensors unchanged by a negative refresh."""
        if stable is not None:
            return ShardBatch(
                stable.users,
                stable.positive,
                self.negative.to(device),
                stable.adjacency,
            )
        return ShardBatch(
            self.users.to(device),
            self.positive.to(device),
            self.negative.to(device),
            self.adjacency.to(device),
        )


class EpochSampler:
    def __init__(self, data, config, *, split_state=None):
        self.data, self.config = data, config
        self.n_spots, self.n_peaks = data.X.shape
        degrees = data.X.getnnz(axis=1)
        if np.any(degrees == self.n_peaks):
            raise ValueError("no negative support for an all-positive spot")
        rows, cols = data.X.nonzero()
        if self.n_spots * self.n_peaks <= config.dense_mask_mb * 1024**2:
            self.mask = np.zeros(data.X.shape, dtype=bool)
            self.mask[rows, cols] = True
            self.keys = None
        else:
            self.mask = None
            self.keys = np.sort(rows.astype(np.int64) * self.n_peaks + cols)
        if split_state is not None:
            required=('train_positive','test_triplets','initial_indptr','initial_indices')
            if any(name not in split_state for name in required):raise ValueError('incomplete saved split')
            if split_state.get('sampling_config',sampling_config(config))!=sampling_config(config):
                raise ValueError('saved split sampling configuration differs')
            if any(not np.issubdtype(np.asarray(split_state[name]).dtype,np.integer) for name in required):
                raise ValueError('saved split indices must be integer arrays')
            self.train_edges=np.asarray(split_state['train_positive'],dtype=np.int64).copy()
            self.test_triplets=np.asarray(split_state['test_triplets'],dtype=np.int64).copy()
            if self.train_edges.ndim!=2 or self.train_edges.shape[0]!=2 or not self.train_edges.size:
                raise ValueError('invalid saved training positive edges')
            if self.test_triplets.ndim!=2 or self.test_triplets.shape[0]!=3:
                raise ValueError('invalid saved test triplets')
            for values in (self.train_edges,self.test_triplets):
                if values.size and (values.min()<0 or values[0].max()>=self.n_spots or values[1:].max()>=self.n_peaks):
                    raise ValueError('saved split index outside node axes')
            train_keys=self.train_edges[0]*self.n_peaks+self.train_edges[1]
            test_keys=np.unique(self.test_triplets[0]*self.n_peaks+self.test_triplets[1])
            observed=np.sort(rows.astype(np.int64)*self.n_peaks+cols)
            if (len(np.unique(train_keys))!=len(train_keys) or np.intersect1d(train_keys,test_keys).size or
                    not np.array_equal(np.union1d(train_keys,test_keys),observed)):
                raise ValueError('saved split does not match observed positive coverage')
            if self.test_triplets.size and self._observed(self.test_triplets[0],self.test_triplets[2]).any():
                raise ValueError('saved test negative is observed')
            # Compiled splits are user-grouped. Regroup an external interleaved
            # split once, stably, rather than scanning every edge for every spot.
            users = self.train_edges[0]
            if np.any(users[1:] < users[:-1]):
                peaks = self.train_edges[1, np.argsort(users, kind="stable")]
            else:
                peaks = self.train_edges[1]
            offsets = np.r_[np.int64(0), np.cumsum(np.bincount(users, minlength=self.n_spots))]
            self.fixed = [peaks[offsets[u]:offsets[u + 1]].copy() for u in range(self.n_spots)]
            pointer=np.asarray(split_state['initial_indptr']);flat=np.asarray(split_state['initial_indices'])
            if len(pointer)!=self.n_spots*config.shards+1 or pointer[0]!=0 or pointer[-1]!=len(flat) or (np.diff(pointer)<0).any():
                raise ValueError('invalid saved initial shard boundaries')
            self.initial=[]
            for u in range(self.n_spots):
                chunks=[flat[pointer[u*config.shards+s]:pointer[u*config.shards+s+1]].copy()
                        for s in range(config.shards)]
                if not np.array_equal(np.sort(np.concatenate(chunks)),np.arange(len(self.fixed[u]))):
                    raise ValueError('initial shard split is not a permutation of training positives')
                self.initial.append(chunks)
        else:
            seeds = np.random.SeedSequence(config.seed).spawn(self.n_spots)
            indptr, indices = data.X.indptr, data.X.indices
            fixed = []
            initial = []
            te = []
            for u in range(self.n_spots):
                rng = np.random.default_rng(seeds[u])
                # Slice the CSR row directly; getrow() copies a whole sparse matrix per spot.
                pos = indices[indptr[u]:indptr[u + 1]].astype(np.int64, copy=True)
                rng.shuffle(pos)
                n = len(pos) if config.tr >= 1 - 1e-12 else int(len(pos) * config.tr)
                train, test = pos[:n], pos[n:]
                fixed.append(train)
                initial.append(
                    np.array_split(rng.permutation(n), config.shards)
                    if n
                    else [np.empty(0, dtype=np.int64)] * config.shards
                )
                if len(test):
                    need = len(test) * config.pn_r
                    candidates = []
                    attempt = 0
                    while len(candidates) < need:
                        draws = rng.integers(0, self.n_peaks, size=need * 2, dtype=np.int64)
                        valid = draws[~self._observed(np.full(len(draws), u, dtype=np.int64), draws)]
                        candidates.extend(valid.tolist())
                        attempt += 1
                        if attempt > 1000:
                            allowed = np.setdiff1d(np.arange(self.n_peaks), pos, assume_unique=True)
                            candidates.extend(rng.choice(allowed, size=need - len(candidates)).tolist())
                    te.append(
                        np.array(
                            [np.full(need, u), np.tile(test, config.pn_r), np.array(candidates[:need])],
                            dtype=np.int64,
                        )
                    )
            self.fixed = fixed
            self.initial = initial
            counts = np.fromiter((len(ps) for ps in fixed), dtype=np.int64, count=self.n_spots)
            self.train_edges = np.stack(
                (
                    np.repeat(np.arange(self.n_spots, dtype=np.int64), counts),
                    np.concatenate(fixed),
                )
            )
            if self.train_edges.size == 0:
                raise ValueError("split has no training positive edges")
            self.test_triplets = np.concatenate(te, axis=1) if te else np.empty((3, 0), dtype=np.int64)
        self.last_reshard_epoch = 0
        self.last_negative_epoch = 0
        self._cache_key = None
        self._batches = None
        self._raw_key = None
        self._raws = None
        self._expanded_key = None
        self._expanded = None
        self._adj_key = None
        self._adj = {}

    def _observed(self, users, peaks):
        if self.mask is not None:
            return self.mask[users, peaks]
        query = users * self.n_peaks + peaks
        index = np.searchsorted(self.keys, query)
        return (index < len(self.keys)) & (
            self.keys[np.minimum(index, len(self.keys) - 1)] == query
        )

    def at_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("epoch must be nonnegative integer")
        c = self.config
        reshard = (epoch // c.reshard_every) * c.reshard_every if c.reshard_every else 0
        neg_epoch = max(
            reshard, (epoch // c.negative_refresh) * c.negative_refresh if c.negative_refresh else 0
        )
        key = (reshard, neg_epoch)
        if key == self._cache_key:
            return self._batches
        raws = self._shard_raws(reshard)
        expanded = self._expanded_raws(reshard, raws)
        shard_seeds = np.random.SeedSequence([c.seed, 0x9E3779B9, neg_epoch]).spawn(c.shards)
        tasks = [(*item, shard_seeds[item[0]]) for item in expanded]
        if len(tasks) > 1:
            with ThreadPoolExecutor(
                max_workers=min(_MAX_NEGATIVE_SAMPLING_WORKERS, len(tasks))
            ) as executor:
                sampled = list(executor.map(self._sample_negative, tasks))
        else:
            sampled = [self._sample_negative(task) for task in tasks]
        result = [
            ShardBatch(
                users_tensor,
                positive_tensor,
                torch.from_numpy(negative),
                self._shard_adjacency(shard, reshard, raw),
            )
            for shard, raw, users_tensor, positive_tensor, negative in sampled
        ]
        self._cache_key = key
        self._batches = result
        self.last_reshard_epoch = reshard
        self.last_negative_epoch = neg_epoch
        return result

    def _sample_negative(self, task):
        shard, raw, users, users_tensor, positive_tensor, seed = task
        rng = np.random.default_rng(seed)
        negative = rng.integers(0, self.n_peaks, size=len(users), dtype=np.int64)
        collision = self._observed(users, negative)
        attempt = 0
        while collision.any():
            indices = np.flatnonzero(collision)
            draw = rng.integers(0, self.n_peaks, size=len(indices), dtype=np.int64)
            negative[indices] = draw
            collision[indices] = self._observed(users[indices], draw)
            attempt += 1
            if attempt > 1000:
                for u in np.unique(users[collision]):
                    index = np.flatnonzero(collision & (users == u))
                    observed = self.data.X.getrow(u).indices
                    allowed = np.setdiff1d(
                        np.arange(self.n_peaks), observed, assume_unique=True
                    )
                    negative[index] = rng.choice(allowed, size=len(index))
                    collision[index] = False
        return shard, raw, users_tensor, positive_tensor, negative

    def _expanded_raws(self, reshard, raws):
        """Expanded users/positives are invariant between negative refreshes."""
        if self._expanded_key == reshard:
            return self._expanded
        expanded = []
        for shard, raw in enumerate(raws):
            if raw is None:
                continue
            users = np.repeat(raw[0], self.config.pn_r)
            positive = np.repeat(raw[1], self.config.pn_r)
            expanded.append(
                (
                    shard,
                    raw,
                    users,
                    torch.from_numpy(users),
                    torch.from_numpy(positive),
                )
            )
        self._expanded_key, self._expanded = reshard, expanded
        return expanded

    def _shard_raws(self, reshard):
        """Per-shard (user, positive) local indices for one reshard epoch.

        A reshard changes only every ``reshard_every`` epochs while negatives are
        refreshed every ``negative_refresh`` epochs, so this split is reused across
        many negative refreshes. Every RNG stream is unchanged, therefore the drawn
        batches are identical to rebuilding the split from scratch each time.
        """
        if self._raw_key == reshard:
            return self._raws
        c = self.config
        user_seeds = (
            np.random.SeedSequence([c.seed, 0xABCDEF01, reshard]).spawn(self.n_spots)
            if reshard
            else None
        )
        chunks = []
        for u, ps in enumerate(self.fixed):
            chunks.append(
                np.array_split(np.random.default_rng(user_seeds[u]).permutation(len(ps)), c.shards)
                if reshard
                else self.initial[u]
            )
        raws = []
        user_ids = np.arange(self.n_spots, dtype=np.int64)
        for s in range(c.shards):
            indices = [chunk[s] for chunk in chunks]
            counts = np.fromiter(
                (len(index) for index in indices), dtype=np.int64, count=self.n_spots
            )
            raws.append(
                np.stack(
                    (
                        np.repeat(user_ids, counts),
                        np.concatenate(
                            [positive[index] for positive, index in zip(self.fixed, indices)]
                        ),
                    )
                )
                if counts.sum()
                else None
            )
        self._raw_key, self._raws = reshard, raws
        self._expanded_key, self._expanded = None, None
        self._adj_key, self._adj = None, {}
        return raws

    def _shard_adjacency(self, shard, reshard, raw):
        """Normalized adjacency for one shard; depends only on the positive split."""
        if self._adj_key != reshard:
            self._adj_key, self._adj = reshard, {}
        if shard not in self._adj:
            self._adj[shard] = bipartite_graph(raw, self.n_spots, self.n_peaks)
        return self._adj[shard]

    def state_dict(self):
        return {
            "last_reshard_epoch": self.last_reshard_epoch,
            "last_negative_epoch": self.last_negative_epoch,
            "mask_backend": "dense" if self.mask is not None else "sorted_edges",
        }

    def split_state(self):
        chunks=[chunk for row in self.initial for chunk in row]
        sizes=np.fromiter((len(c) for c in chunks),dtype=np.int64,count=len(chunks))
        return {'train_positive':self.train_edges,'test_triplets':self.test_triplets,
                'initial_indptr':np.r_[np.int64(0),np.cumsum(sizes)],
                'initial_indices':np.concatenate(chunks).astype(np.int64,copy=False)}


def prepare_training(network,config,output):
    """Save the fixed split, initial shards and training graph as TrainingInputs.

    Accept a ModelNetwork handle or directory and a TrainConfig. ``output``
    must be unused. This preparation stage does not optimize model parameters.
    """
    from pathlib import Path
    from .artifacts import ModelNetwork,TrainingInputs,stage_metadata,save_tensor_graph,validation_scope
    from .utils import artifact_writer,write_json,protect_output
    from .graph import bipartite_graph
    if Path(output).exists():raise FileExistsError(output)
    with validation_scope():
        network=ModelNetwork.load(network)
        if network.manifest['kind']!='model_network':
            raise ValueError('migrate a legacy prepared artifact before preparing training inputs')
        output=protect_output(output,[network.path]);data=network.load_data()
        sampler=EpochSampler(data,config)
        metadata=stage_metadata(output,upstreams={'network':network},
            peak_ids=data.peak_ids,spot_ids=data.spot_ids,motif_ids=data.dna_ids,
            data_fingerprint=network.metadata['data_fingerprint'],sampling_config=sampling_config(config),
            protocol='copy1_fixed_split_initial_shards_v1',n_train_positive=int(sampler.train_edges.shape[1]),
            n_test_triplets=int(sampler.test_triplets.shape[1]),shape=list(data.X.shape))
        metadata['caller_data_fingerprint']=network.metadata.get('caller_data_fingerprint')
        with artifact_writer(output,'training_inputs',metadata) as stage:
            np.savez_compressed(stage/'split.npz',**sampler.split_state())
            save_tensor_graph(stage/'full_train_adjacency.npz',bipartite_graph(sampler.train_edges,*data.X.shape))
            write_json(stage/'sampling.json',metadata['sampling_config'])
        return TrainingInputs.load(output)
