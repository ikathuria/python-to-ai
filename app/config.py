class Config(object):
    DEBUG = True
    TESTING = False


class DevelopmentConfig(Config):
    SECRET_KEY = "this-is-a-super-secret-key"


config = {
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'production': DevelopmentConfig
}

# Enter your Open API Key here
OPENAI_API_KEY = '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'
