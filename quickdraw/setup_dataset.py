from utils import get_dataset
from utils import load_df_from_json
from utils import export_df_to_hd5


get_dataset()
all_df = load_df_from_json()
export_df_to_hd5(all_df)
