import pandas as pd


csv_file = 'CPRM_AutoLOan_OnlineAutoLoanData.csv'
df = pd.read_csv(csv_file)

def convert_row_to_dict(row, next_state_values,done):
    state_columns = ["Tier","Primary_FICO","Type","Term","New_Rate","Used_Rate","Amount_Approved","CarType","onemonth","months","partnerbin","CarType_id"]
    state_values = [row[col] for col in state_columns]
    return {
        'state': state_values,
        'action': [row['rate']],
        'reward': float(row['Term']),
        'next_state': next_state_values,
        'done': done
    }

formatted_dataset = []
for idx, row in enumerate(df.iterrows()):
    _, row_data = row
    if idx < len(df) - 1:
        next_row_data = df.iloc[idx + 1]
        next_state_columns = ["Tier","Primary_FICO","Type","Term","New_Rate","Used_Rate","Amount_Approved","CarType","onemonth","months","partnerbin","CarType_id"]
        next_state_values = [next_row_data[col] for col in next_state_columns]
    else:
        next_state_values = None  # or set it to a default value
    if(idx < 10000):
        done = 0.0
    else:
        done = 1.0
    formatted_dataset.append(convert_row_to_dict(row_data, next_state_values,done))
from d3rlpy.dataset import MDPDataset

states = []
actions = []
rewards = []
next_states = []
dones = []

for data in formatted_dataset:
    states.append(data['state'])
    actions.append(data['action'])
    rewards.append(data['reward'])
    next_states.append(data['next_state'])
    dones.append(data['done'])
dataset = MDPDataset(
    states=states,
    actions=actions,
    rewards=rewards,
    next_states=next_states,
    terminals=dones
)