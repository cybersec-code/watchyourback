#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Entity module: provides Entity objects to be used in explorations."""

import logging
import pandas as pd
try:
    from lib.tags import is_service_owner
    from lib.blockchain_feature_extraction import extract_features_range
except ModuleNotFoundError:
    from tags import is_service_owner
    from blockchain_feature_extraction import extract_features_range

class Entity:
    """ Set of properties of a Bitcoin Entity """

    def __init__(self, cid, owner='', ctags='', art_c=None, cm=None, height=0, ignore_wd=True):

        if not (art_c or cm):
            return

        self.cid = cid
        self.cids = {cid}

        if art_c:
            self.cluster = art_c[cid]
            self.addrs = self.cluster['addrs']
            self.output_txes = self.cluster['output_txes']
            if ignore_wd:
                self.input_txes = set()
            else:
                self.input_txes = self.cluster['input_txes']
        else:
            self.cluster = cm.clusters()[cid]
            self.addrs = set(self.cluster.addresses.to_list())
            self.output_txes = set(
                self.cluster.addresses.select(
                    lambda a: a.output_txes.where(
                        lambda t: t.block.height <= height
                    )
                )
            )
            if ignore_wd:
                self.input_txes = set()
            else:
                self.input_txes = set(
                    self.cluster.addresses.select(
                        lambda a: a.input_txes.where(
                            lambda t: t.block.height <= height
                        )
                    )
                )

        self.size = len(self.addrs)
        self.owner = owner
        self.ctags = ctags
        self.csize = -1
        self.output_txes_count = len(self.output_txes)
        self.output_txes_fees = 0
        self.outputs = set()
        self.outputs_count = 0
        self.input_txes_count = len(self.input_txes)
        self.input_txes_fees = 0
        self.inputs = set()
        self.inputs_count = 0
        self.coinbase_txes = 0
        self.coinjoin_txes_out = set()
        self.coinjoin_txes_out_count = 0
        self.coinjoin_txes_in = set()
        self.coinjoin_txes_in_count = 0
        self.deposited = 0
        self.withdrawn = 0
        self.ts_first_tx = None
        self.ts_last_tx = None
        self.height = height
        self.d_clusters = set()
        self.w_clusters = set()
        self.prob = None
        self.service = is_service_owner(owner)

    def classify(self, classifier, chain, cm):
        """
        Classify this Entity. The results are saved in prob and service vars.
        """
        if not classifier or self.service:
            return
        addrs = {a: None for a in self.addrs}
        vectors = extract_features_range(addrs, chain, cm, self.height)
        dataset = pd.DataFrame(data=vectors, columns=vectors[0].keys())
        X = dataset.drop(['label'], axis=1)
        cresult = classifier.predict_proba(X)
        self.prob = sum([r[0] for r in cresult]) / len(cresult)
        self.service = 'EC' if self.prob > 0.5 else None
        msg = "EC AVGPROB %s: %s => %s"
        logging.debug(msg, self.cid, self.prob, self.service)

    def update_addrs(self, addrs):
        """
        Update set of addresses of this Entity. Update the size also.
        size is not the same as csize (MI cluster size).
        """
        self.addrs.update(addrs)
        self.size = len(self.addrs)

    def update_outputs(self, outputs):
        """
        Update the set of outputs of this Entity.
        """
        self.outputs.update(outputs)
        self.outputs_count = len(self.outputs)

    def update_inputs(self, inputs):
        """
        Update the set of inputs of this Entity.
        """
        self.inputs.update(inputs)
        self.inputs_count = len(self.inputs)

    def update_txes_ts(self):
        """
        Update the set of txes of this Entity. Update also ts_first/last_tx.
        """
        txes = self.input_txes.union(self.output_txes)
        timestamps = {t.block.timestamp for t in txes}
        self.ts_first_tx = min(timestamps)
        self.ts_last_tx = max(timestamps)

    def update_output_txes(self, out_txes):
        """
        Update the set of output txes of this Entity.
        """
        self.output_txes.update(out_txes)
        self.output_txes_count = len(self.output_txes)
        self.update_txes_ts()

    def update_input_txes(self, in_txes):
        """
        Update the set of input txes of this Entity.
        """
        self.input_txes.update(in_txes)
        self.input_txes_count = len(self.input_txes)
        self.update_txes_ts()

    def update_output_txes_fees(self, fees):
        """
        Update the set of output txes fees of this Entity.
        """
        self.output_txes_fees = fees

    def update_input_txes_fees(self, fees):
        """
        Update the set of input txes fees of this Entity.
        """
        self.input_txes_fees = fees

    def update_coinjoin_txes_out(self, cj_txes_out):
        """
        Update the set of coinjoin output txes of this Entity.
        """
        self.coinjoin_txes_out.update(cj_txes_out)
        self.coinjoin_txes_out_count = len(self.coinjoin_txes_out)

    def update_coinjoin_txes_in(self, cj_txes_in):
        """
        Update the set of coinjoin input txes of this Entity.
        """
        self.coinjoin_txes_in.update(cj_txes_in)
        self.coinjoin_txes_in_count = len(self.coinjoin_txes_in)

    def update_info(self, owner, ctags, csize):
        """
        Update the attribution information of this entity.
        """
        self.owner = owner
        self.ctags = ctags
        self.csize = csize

    def has_addr(self, addr):
        """
        Determine if this Entity contains an address
        """
        return addr in self.addrs

    def balance(self):
        """
        Determine the total balance of this Entity
        """
        return self.deposited - self.withdrawn - self.input_txes_fees

    def txes(self):
        """
        Return the set of txes of this Entity
        """
        txes = self.input_txes.union(self.output_txes)
        return len(txes), txes

    def coinjoin_txes(self):
        """
        Return the set of coinjoin txes of this Entity
        """
        cj = self.coinjoin_txes_in.union(self.coinjoin_txes_out)
        return len(cj), cj

    def __str__(self):
        cids = [str(c) for c in sorted(self.cids)]
        return f"{self.cid}:{','.join(cids)}:{self.owner}"

    def __repr__(self):
        return f"{self.cid}:{self.owner}"
