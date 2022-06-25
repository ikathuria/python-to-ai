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
