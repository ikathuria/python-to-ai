class Config(object):
    DEBUG = True
    TESTING = False


class DevelopmentConfig(Config):
    SECRET_KEY = "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"


config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}

OPENAI_API_KEY = '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'
