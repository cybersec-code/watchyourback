""" Vector provides objects to be used with the classifier """

from datetime import datetime
from collections import defaultdict

class Vector:
    """ Feature vector of a single bitcoin address """

    def __init__(self, addr):
        self.addr = addr
        self.cluster = -1
        self.clust_addrs = 0
        self.eq_addrs = 0
        self.txes_count = 0
        self.out_txes_count = 0
        self.in_txes_count = 0
        self.txes_outputs = 0
        self.in_txes_outputs = 0
        self.out_txes_outputs = 0
        self.txes_inputs = 0
        self.in_txes_inputs = 0
        self.out_txes_inputs = 0
        self.outputs = 0
        self.inputs = 0
        self.utxos = 0
        self.inputs_age = 0
        self.tx_sizes = []
        self.tx_weights = []
        self.tx_fees = []
        self.txes_coinbase = 0
        self.txes_coinjoin = 0
        self.txes_coinjoin_out = 0
        self.txes_coinjoin_in = 0
        self.withdrawn = 0
        self.deposited = 0
        self.ts_first_out = None
        self.ts_last_out = None
        self.ts_first_in = None
        self.ts_last_in = None
        self.activity_days = set()
        self.activity_days_out = set()
        self.activity_days_in = set()
        self.txes_years_out = defaultdict(int)
        self.txes_years_in = defaultdict(int)
        self.num_txes_same_as_change = 0
        self.is_out_tx = False
        self.is_in_tx = False

    def balance(self):
        """
        Address balance (deposited - withdrawn)
        """
        return self.deposited - self.withdrawn

    def activity(self):
        """
        Number of days this address has been active
        """
        return len(self.activity_days)

    def activity_d(self):
        """
        Number of days this address has been active (deposits)
        """
        return len(self.activity_days_out)

    def activity_w(self):
        """
        Number of days this address has been active (withdrawals)
        """
        return len(self.activity_days_in)

    def tx_ratio(self):
        """
        Ratio of withdrawals per deposits
        """
        return self.in_txes_count/self.out_txes_count

    def d_per_tx(self):
        """
        Average BTCs deposited in each deposit
        """
        return self.deposited/self.out_txes_count

    def w_per_tx(self):
        """
        Average BTCs withdrawn in each withdrawal (or -1)
        """
        return self.withdrawn/self.in_txes_count if self.in_txes_count else -1

    def addr_as_change(self):
        """
        Ratio of number of txes using this addr in a self-change tx
        """
        return self.num_txes_same_as_change / self.txes_count

    def ts_first(self):
        """
        Timestamp of the first deposit to this addr
        """
        return min(self.ts_first_out, self.ts_last_out)

    def datetime_first(self):
        """
        Date of the first deposit to this addr
        """
        return datetime.utcfromtimestamp(self.ts_first())

    def ts_last(self):
        """
        Timestamp of the last tx of this addr
        """
        all_ts = [self.ts_first_out, self.ts_last_out, self.ts_first_in, self.ts_last_in]
        return max([ts for ts in all_ts if ts])

    def datetime_last(self):
        """
        Date of the last tx of this addr
        """
        return datetime.utcfromtimestamp(self.ts_last())

    def lifetime(self):
        """
        Number of seconds between the first and last tx of this addr
        """
        return self.ts_last() - self.ts_first()

    def total_days(self):
        """
        Number of days between the first and last tx of this addr
        """
        return int(self.lifetime() / (3600*24))

    def daily_d_rate(self):
        """
        Average number of daily deposits to this addr
        """
        return self.out_txes_count/self.total_days() if self.total_days() else -1

    def daily_w_rate(self):
        """
        Average number of daily withdrawals from this addr
        """
        return self.in_txes_count/self.total_days() if self.total_days() else -1

    def idle_days(self):
        """
        Number of days this addr has no activity
        """
        return self.total_days() - self.activity()

    def timespan_out(self):
        """
        Timespan bewteen the first and last depost
        """
        return self.ts_last_out - self.ts_first_out

    def timespan_in(self):
        """
        Timespan bewteen the first and last withdrawal
        """
        return self.ts_last_in - self.ts_first_in if self.ts_first_in else 0

    def yearly_d_txes(self):
        """
        Average number of yearly deposits
        """
        return sum(self.txes_years_out.values())/len(self.txes_years_out)

    def yearly_w_txes(self):
        """
        Average number of yearly withdrawals
        """
        return sum(self.txes_years_in.values())/len(self.txes_years_in) \
                if self.txes_years_in else -1

    def __str__(self):
        return self.addr

    def __repr__(self):
        return self.addr
