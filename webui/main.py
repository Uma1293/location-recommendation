from flask import Flask, render_template, request, flash
import numpy
import pandas
from scipy import sparse
import networkx as nx
from wtforms import Form, IntegerField, TextField, TextAreaField, validators, StringField, SubmitField
import gc

load_vars = False
user_list = None
loc_list = None
user_idx = None
loc_idx = None
user_item_csr = None
DG = None
P = None
Q = None

def get_user_item():
  global user_list
  global loc_list
  global user_item_csr
  # user-item matrix generated in earlier file
  user_item_sparse = sparse.load_npz('generated_data/user_item_matrix.npz')
  user_item_csr = user_item_sparse.tocsr()
  
  user_loc_f = numpy.load('generated_data/user_item_matrix_columns.npz')
  user_list = list(user_loc_f['arr_0'])
  loc_list = list(user_loc_f['arr_1'])

def get_user_loc():
  global user_idx
  global loc_idx
  
  # create user_idx and loc_idx
  user_idx = {}
  idx = 0
  for u in user_list:
      user_idx[u] = idx
      idx += 1
  
  loc_idx = {}
  idx = 0
  for l in loc_list:
      loc_idx[l] = idx
      idx += 1
  #
  
def get_P_Q():
  global P
  global Q
  P_f = numpy.load('generated_data/user_item_decomposed_P.npz')
  Q_f = numpy.load('generated_data/user_item_decomposed_Q.npz')
  P = P_f['arr_0']
  Q = Q_f['arr_0']
  
def get_DG():
  global DG
  # read the friends network data
  friends_f = numpy.load('generated_data/friendship_data.npz')
  friends_data = list(zip(friends_f['arr_0'],friends_f['arr_1']))
  
  DG = nx.DiGraph()
  
  # add nodes based on the user-list
  for u in user_list:
      DG.add_node(u)
      
  # add edges: friends based on friend data
  for e in friends_data:
      DG.add_edge(e[0],e[1])


def get_friends(userid,depth_limit=1):
    friends_e = nx.bfs_edges(DG,userid,depth_limit=depth_limit)
    friends = []
    for e in friends_e:
        friends.append(e[1])
        if len(friends) >= 10:
          break
    
    return friends
    
def has_friend_visited(friends,usern,locn):
    # we use this function to check if a friend 
    # of userid has visited this locid. 
    userid = user_list[usern]
    for friendid in friends:
        friendn = user_idx[friendid]
        if user_item_csr[friendn,locn] != 0:
            return True
    
    return False  
    
def recommend_locations_base(userid):
    usern = user_idx[userid]
    
    predictions = numpy.dot(P[usern,:][numpy.newaxis],Q.T) # we get the predictions by dot product of P.Q [userid]
    predictions = predictions[0] #drop the two-dimensions
    predictions_idx = [x for _,x in sorted(zip(predictions,range(len(predictions))),reverse=True)]

    recommendations = numpy.array(loc_list)[predictions_idx]
    return recommendations
    
def recommend_locations_without_friends(userid,K=10):
    recommendations = recommend_locations_base(userid)
    
    # filter the recommended locations based on:
    # 1. it should not be visited by userid

    filtered_not_visited = []
    n = 0
    usern = user_idx[userid]
    while len(filtered_not_visited) < K and n < len(recommendations):
        locn = loc_idx[recommendations[n]]
        # 1:
        if user_item_csr[usern,locn] == 0:
            filtered_not_visited.append(recommendations[n])

        n += 1

    return filtered_not_visited

def recommend_locations_with_friends(userid,K=10):
    recommendations = recommend_locations_base(userid)
    
    # filter the recommended locations based on:
    # 1. it should not be visited by userid
    # 2. it should be visited by one of his friends
    friends = get_friends(userid)
    filtered_not_visited = []
    filtered_not_visited_and_friend_visited = []
    n = 0
    usern = user_idx[userid]
    while len(filtered_not_visited_and_friend_visited) < K and n < len(recommendations):
        locn = loc_idx[recommendations[n]]
        # 1 and 2:
        if user_item_csr[usern,locn] == 0:
            filtered_not_visited.append(recommendations[n])
        
        if user_item_csr[usern,locn] == 0 and has_friend_visited(friends,usern,locn):
            filtered_not_visited_and_friend_visited.append(recommendations[n])

        n += 1
    
    # its possible that user has no friends, so use the base
    # recommender for these cases
    if len(filtered_not_visited_and_friend_visited) == 0:
        return filtered_not_visited[0:K]
    else:
        return filtered_not_visited_and_friend_visited


app = Flask(__name__)

# Model
class InputForm(Form):
    ip = IntegerField(validators=[validators.InputRequired()])

class DBForm():
    def __init__(self,ul,uid):
        self.user_list = ul
        self.user_id = uid
        self.user_n = None
    
        if ul is not None:
          self.max_users = len(ul)
        else:
          self.max_users = None
        
        # recommendation results
        self.recommended_loc_w_f = None
        self.recommended_loc_wo_f = None
        self.friends = None
        

@app.route("/",methods=['GET', 'POST'])
def home():
    global load_vars
    if load_vars == False:
      get_user_item()
      get_user_loc()
      get_P_Q()
      get_DG()
      load_vars = True

    form = InputForm(request.form)
    db = DBForm(user_list,None)
    
    if request.method == 'POST' and form.validate():
        db.user_id = user_list[form.ip.data%len(user_list)]
        db.user_n = form.ip.data
        
        db.friends = get_friends(db.user_id)
        db.recommended_loc_w_f = recommend_locations_with_friends(db.user_id)
        db.recommended_loc_wo_f = recommend_locations_without_friends(db.user_id)
        #print(len(db.friends))
        #print(len(db.recommended_loc_w_f))
        #print(len(db.recommended_loc_wo_f))
    
    return render_template("view.html",form=form,db=db)

if __name__ == "__main__":
    app.run(debug=True)
    
