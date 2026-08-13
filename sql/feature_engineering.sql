SELECT 
    transaction_id,
    time_seconds,
    amount,
    is_fraud,
    COUNT(*) OVER (
        ORDER BY time_seconds 
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS txn_count_last_hour,
    
    AVG(amount) OVER (
        ORDER BY time_seconds 
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_hour,
    
    time_seconds - LAG(time_seconds) OVER (ORDER BY time_seconds) AS time_since_last_txn

FROM transactions
ORDER BY time_seconds
LIMIT 20;
