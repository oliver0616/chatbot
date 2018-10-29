import os
from StringIO import StringIO

from bson.json_util import dumps
from bson.json_util import loads
from bson.objectid import ObjectId
from flask import Blueprint, request, Response
from flask import abort
from flask import current_app as app
from flask import send_file

from app.commons import build_response
from app.commons.utils import update_document

login_blueprint = Blueprint('login_blueprint', __name__,
                    url_prefix='/login')

@login_blueprint.route('/', methods=['POST'])
def create_login():
    content = request.get_json(silent=True)

    username = content["username"]
    password = content["password"]

    if username == "test" and password == "p":
        return build_response.build_json({"status":"ok"})
    else:
        return build_response.build_json({"status": "error"})

