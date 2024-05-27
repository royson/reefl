from flwr.server.client_manager import SimpleClientManager as FlwrSimpleClientManager
from typing import Optional, List, Any
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from src.utils import get_func_from_config
import random
from flwr.common.logger import log
from logging import INFO

class BudgetClientManager(FlwrSimpleClientManager):
    '''
    Based on Flower's SimpleClientManager. 

    Available clients are based on budget
    '''
    def __init__(self, ckp, *args, max_exit_budget=None, **kwargs):
        # max_exit_budget starts from 1 to max_exit
        super().__init__(*args, **kwargs)
        no_of_exits = ckp.config.models.net.args.no_of_exits
        no_of_clients = ckp.config.simulation.num_clients
        client_max_exit_layers = {str(i): i % no_of_exits for i in range(no_of_clients)}
        self.no_of_clients = no_of_clients
        if max_exit_budget is None:
            # use max exit
            max_exit_budget = no_of_exits
        assert max_exit_budget > 0 and max_exit_budget <= no_of_exits

        self.list_of_clients = [cid for cid in range(no_of_clients) if (client_max_exit_layers[str(cid)] + 1) >= max_exit_budget]
        print(f"List of client CIDs with sufficient budget of {max_exit_budget}: {self.list_of_clients}")
        assert self.list_of_clients, 'No available clients'        
        
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion: Optional[Criterion] = None,
        all_available: Any = False
    ) -> List[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.

        if all_available:
            sampled_cids = random.sample(list(self.clients.keys()), num_clients)
            return [self.clients[str(cid)] for cid in sampled_cids]

        if num_clients > len(self.list_of_clients):
            # case where there is insufficient clients
            num_clients = len(self.list_of_clients)

        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)

        sampled_cids = random.sample(self.list_of_clients, num_clients)
        return [self.clients[str(cid)] for cid in sampled_cids]