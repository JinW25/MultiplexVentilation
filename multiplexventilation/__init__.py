from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from multiplexventilation.config import Config


db = SQLAlchemy()


def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(Config)
    db.init_app(app)

    with app.app_context():
        # Import parts of the core Flask app
        from multiplexventilation.PCmode.routes import PCmode
        from multiplexventilation.main.routes import main
        app.register_blueprint(PCmode)
        app.register_blueprint(main)

        return app
