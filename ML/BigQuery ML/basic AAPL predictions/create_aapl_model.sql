CREATE OR REPLACE MODEL `animated-rhythm-324612.AI4F.aapl_model`
    OPTIONS
    (model_type = 'linear_reg',
    input_label_cols = ['close'],
    data_split_method = 'seq',
    data_split_eval_fraction=0.3,
    data_split_col='date') as 
SELECT 
    date,
    close,
    day_prev_close,
    trend_3_day
FROM `animated-rhythm-324612.AI4F.model_data`