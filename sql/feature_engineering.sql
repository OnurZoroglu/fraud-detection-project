-- Her işlem için, aynı işlemden önceki 1 saat içinde kaç işlem yapılmış (window function)
SELECT 
    transaction_id,
    time_seconds,
    amount,
    is_fraud,
    COUNT(*) OVER (
        ORDER BY time_seconds 
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS txn_count_last_hour,
    
    -- Son 1 saatteki ortalama işlem tutarı
    AVG(amount) OVER (
        ORDER BY time_seconds 
        RANGE BETWEEN 3600 PRECEDING AND CURRENT ROW
    ) AS avg_amount_last_hour,
    
    -- Bu işlemin, önceki işlemden ne kadar zaman sonra geldiği
    time_seconds - LAG(time_seconds) OVER (ORDER BY time_seconds) AS time_since_last_txn

FROM transactions
ORDER BY time_seconds
LIMIT 20;