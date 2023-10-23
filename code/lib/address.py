""" Address object to be used during the exploration """

from lib import blockchain_feature_extraction as bfe

class Address:
    ''' Bitcoin address and analysis-related information '''

    def __init__(self, addr, txes, cid=None, owner=None, tag=None, ctags=None,
                 csize=None, step=None, pred=-1.0, op=None, seed=False,
                 exch=False, blocklist=False, service=False, ticker='btc'):
        self.addr = addr
        self.txes = txes
        self.name = bfe.addr_to_string(addr)
        self.cid = cid
        self.owner = owner
        self.tag = tag if tag else ''
        self.ctags = ctags if ctags else ''
        self.csize = csize
        self.step = step
        self.pred = pred
        self.op = op
        self.seed = seed
        self.exch = exch
        self.blocklist = blocklist
        self.service = service
        self.ticker = ticker
        self.fullname = f"{self.ticker}:{self.name}"
        self.fulltag = f"{self.owner}>>{self.tag}" if (owner or tag) else ''

    def update(self, cid=None, owner=None, tag=None, ctags=None, csize=None,
               exch=None, blocklist=None, txes=None, pred=None, op=None):
        """
        Update one or more fields of this address.
        """
        if cid is not None:
            self.cid = cid
        if owner is not None:
            self.owner = owner
        if tag is not None:
            self.tag = tag
        if ctags is not None:
            self.ctags = ctags
        if csize is not None:
            self.csize = csize
        if exch is not None:
            self.exch = exch
        if blocklist is not None:
            self.blocklist = blocklist
        if txes is not None:
            self.txes = txes
        if pred is not None:
            self.pred = pred
        if op is not None:
            self.op = Operation(op['perc'], op['iocs'])

    def visited_txes(self, visited=None):
        """
        Set the deposit/withdrawal txes visited from this address.
        :param visited: Set of visited txes. Only deposit/withdrawal txes in
        the intersection with visited will remain in the graph.
        """
        if visited is None:
            visited = set()
        self.txes['d_txes'] &= visited
        self.txes['w_txes'] &= visited

    def isop(self):
        """
        Determine if this address belongs to the operation explored.
        """
        return self.op.isop if self.op else False

    def __str__(self):
#        TODO return self.fullname
        return self.name

    def __repr__(self):
#        TODO return self.fullname
        return self.name

class Operation:
    """
    Class to manage IOCs from different operations
    """
    def __init__(self, rate=0.0, iocs=None):
        """
        :param float rate: rating of the prediction, between 0 and 1
        :param dict iocs: dict object with the IOCs
        """
        self.rate = rate
        self.iocs = iocs
        self.isop = rate > 0.5

    def __str__(self):
        return f"IOCs: {len(self.iocs)}, rate: {self.rate}"

    def __repr__(self):
        return f"IOCs: {len(self.iocs)}, rate: {self.rate}"
