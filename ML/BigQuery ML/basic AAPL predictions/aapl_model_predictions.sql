SELECT * FROM ML.PREDICT(MODEL `animated-rhythm-324612.AI4F.aapl_model`,(
    SELECT * FROM `animated-rhythm-324612.AI4F.model_data`
    WHERE date >= '2019-01-01'
) )