WITH raw as (
    SELECT 
        date,
        close,
        LAG(close, 1) OVER(ORDER BY date) AS min_1_close,
        LAG(close, 2) OVER(ORDER BY date) AS min_2_close,
        LAG(close, 3) OVER(ORDER BY date) AS min_3_close,
        LAG(close, 4) OVER(ORDER BY date) AS min_4_close,
    FROM `animated-rhythm-324612.AI4F.AAPL10Y`
    ORDER BY date DESC
),
raw_plus_trend as (
    SELECT 
        date,
        close, 
        min_1_close,
        IF (min_1_close - min_2_close > 0, 1, -1) AS min_1_trend,
        IF (min_2_close - min_3_close > 0, 1, -1) AS min_2_trend,
        IF (min_3_close - min_4_close > 0, 1, -1) AS min_3_trend
    FROM raw 
),
ml_data as (
    SELECT 
        date,
        close,
        min_1_close as day_prev_close,
        IF(min_1_trend + min_2_trend + min_3_trend > 0 , 1, -1) as trend_3_day
    FROM raw_plus_trend 
)
SELECT * from ml_data