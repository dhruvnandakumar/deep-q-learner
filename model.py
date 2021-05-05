import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc2_units)
        self.fc4 = nn.Linear(fc2_units, action_size)
        self.dropout = nn.Dropout(0.2) # Dropout is shown to help dqn convergence

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(self.dropout(x)))
        x = F.relu(self.fc3(self.dropout(x)))
        return self.fc4(self.dropout(x))
    
    
    
    
class DuelingQNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed, action_value_size=512, state_value_size=512):
        
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        #-------Construct network for estimating action value--------# 
        act_layers = OrderedDict()
        act_layers['act_fc1'] = nn.Linear(state_size, action_value_size)
        act_layers['act_fc1_act'] = nn.ReLU()
        act_layers['act_fc1_drp'] = nn.Dropout(0.2)
#         act_layers['act_fc2'] = nn.Linear(action_value_size, action_value_size)
#         act_layers['act_fc2_act'] = nn.ReLU()
#         act_layers['act_fc2_drp'] = nn.Dropout(0.2)
        act_layers['act_final'] = nn.Linear(action_value_size, action_size)
        self.actionValueNet = nn.Sequential(act_layers)
        
        #-------Construct network for estimating state value--------# 
        state_layers = OrderedDict()
        state_layers['state_fc1'] = nn.Linear(state_size, action_value_size)
        state_layers['state_fc1_act'] = nn.ReLU()
        state_layers['state_fc1_drp'] = nn.Dropout(0.2)
#         state_layers['state_fc2'] = nn.Linear(action_value_size, action_value_size)
#         state_layers['state_fc2_act'] = nn.ReLU()
#         state_layers['state_fc2_drp'] = nn.Dropout(0.2)
        state_layers['state_final'] = nn.Linear(action_value_size, 1)
        self.stateValueNet = nn.Sequential(state_layers)
        
        
    def forward(self, state):
        action_value = self.actionValueNet(state)
        state_value = self.stateValueNet(state)
        
        return state_value + (action_value - action_value.mean()) #https://towardsdatascience.com/dueling-deep-q-networks-81ffab672751
        
        