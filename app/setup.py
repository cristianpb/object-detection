from setuptools import setup

setup(
        name="flask_app",
        packages=["flask_app"],
        install_requires=[
            'Click==7.0',
            'Flask==1.0.2',
            'itsdangerous==1.1.0',
            'Jinja2==2.10',
            'MarkupSafe==1.1.1',
            'numpy==1.16.2',
            'opencv-python-headless==4.0.0.21',
            'pandas==0.24.2',
            'Pillow==5.4.1',
            'python-dotenv==0.10.2',
            'Werkzeug==0.14.1',
            ]
        )
