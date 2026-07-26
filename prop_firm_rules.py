class PropFirmRules:
    DAILY_LOSS_LIMIT_PCT = 0.05
    MAX_LOSS_LIMIT_PCT = 0.10
    PROFIT_TARGET_PCT = 0.10
    MIN_TRADING_DAYS = 10

    @staticmethod
    def check_daily_loss(start, current):
        return (start - current) / start >= 0.05 if start > 0 else False

    @staticmethod
    def check_max_loss(initial, current):
        return (initial - current) / initial >= 0.10 if initial > 0 else False
