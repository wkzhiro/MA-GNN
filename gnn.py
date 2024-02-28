from fastapi import FastAPI, Depends, Request, HTTPException, Header,BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sqlalchemy import create_engine
from sqlalchemy.sql import text
# import mysql.connector
import pandas as pd

import torch
import numpy as np
from torch.nn import Linear, Embedding, Dropout
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

import os
from os.path import join, dirname
import random
from dotenv import load_dotenv

# .envファイルを読み込む
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path,override=True)

# MySQLデータベースへの接続情報
db_config = {
    "host": os.getenv('DB_HOST'),
    "user": os.getenv('DB_USERNAME'),
    "password": os.getenv('DB_PASSWORD'),
    "database": os.getenv('DB_NAME')
}

# 環境変数の値を表示（確認用）
print(db_config)

# SQLAlchemyで使用するデータベース接続URLを構築
db_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}"

# SQLAlchemy Engineを作成
engine = create_engine(db_url)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "model_droplate0.5.pth"

# FastAPIをインスタンス化
app = FastAPI()

# MySQLに接続
try:
    # connection = mysql.connector.connect(**db_config)
    with engine.connect() as connection:

        # Usersテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM Users")
        result = connection.execute(sql_query)
        # 結果を取得し、データフレームに変換
        users_result = result.fetchall()
        users_df = pd.DataFrame(users_result, columns=[i for i in result.keys()])
        print("Usersテーブルの情報:")
        print(users_df)
        print("\n")

        # Contentテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM Content")
        result = connection.execute(sql_query)
        content_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Contentテーブルの情報:")
        print(content_df)
        print("\n")

        # Tagsテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM Tags")
        result = connection.execute(sql_query)
        tags_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Tagsテーブルの情報:")
        print(tags_df)
        print("\n")

        # Tag_campaignテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM tag_content")
        result = connection.execute(sql_query)
        tag_content_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Tag_contentテーブルの情報:")
        print(tag_content_df)
        print("\n")

        # Cookieテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM Cookie")
        result = connection.execute(sql_query)
        cookie_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Cookieテーブルの情報:")
        print(cookie_df)
        print("\n")

        # Activityテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM activity")
        result = connection.execute(sql_query)
        activity_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Activityテーブルの情報:")
        print(activity_df)
        print("\n")

        # Handling_propertyテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM Handling_property")
        result = connection.execute(sql_query)
        handling_property_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("Handling_propertyテーブルの情報:")
        print(handling_property_df)
        print("\n")

        # User_propertyテーブルのデータを取得し、データフレームに変換
        sql_query = text("SELECT * FROM User_property")
        result = connection.execute(sql_query)
        user_property_df = pd.DataFrame(result.fetchall(), columns=[i for i in result.keys()])
        print("User_propertyテーブルの情報:")
        print(user_property_df)
        print("\n")

except Exception as e:
    print("接続エラー: ",e)


def process_content_data(content_df, tag_content_df, tags_df):
    # コンテンツとタグの紐づけ
    content_tag_df = content_df.merge(tag_content_df, on="content_id", how="left")
    content_tag_df = content_tag_df.fillna("")

    # content_idごとにタグ情報をリストでまとめる
    content_tag_df = content_tag_df.groupby("content_id").agg({
        "title": "first",
        "p_type": "first",
        "url": "first",
        "tag_id": lambda x: x.tolist()
    }).reset_index()

    # 全タグのダミーデータフレームを作成
    all_tags_df = pd.DataFrame({'tag_id': tags_df['tag_id'].unique()})
    all_tags_df['key'] = 0

    # 元のデータフレームにキーを追加して全タグと結合
    content_tag_df['key'] = 0
    merged_df = content_tag_df.merge(all_tags_df, on='key', how='left').drop('key', axis=1)

    # タグの存在をチェックしてダミー変数を設定
    merged_df['dummy'] = merged_df.apply(lambda row: 1 if row['tag_id_y'] in row['tag_id_x'] else 0, axis=1)

    # ダミー変数を元のデータフレームに結合
    dummy_df = merged_df.pivot(index='content_id', columns='tag_id_y', values='dummy').fillna(0).astype(int)
    dummy_df.columns = ['tag' + str(int(col)) for col in dummy_df.columns]

    content_tag_df = pd.concat([content_tag_df.set_index('content_id'), dummy_df], axis=1).reset_index()
    content_tag_df = content_tag_df.drop(columns=["tag_id"])

    # content_idが空白の行を削除
    content_tag_df = content_tag_df[content_tag_df['content_id'] != 0]

    # インデックスをリセット
    content_tag_df = content_tag_df.reset_index(drop=True)
    print(content_tag_df)
    return content_tag_df

def process_user_data(users_df, user_property_df, handling_property_df):
    # userテーブルとhandlingテーブルをuser_idで結合
    user_handling_df = users_df.merge(user_property_df, on="user_id", how="left")
    # NaNを空白文字列に置き換え
    user_handling_df = user_handling_df.fillna("")

    user_handling_df = user_handling_df.groupby("user_id").agg({
        "cookie_id": "first",
        "age": "first",
        "sex": "first",
        "industry": "first",
        "handling_property_id": lambda x: x.tolist()
    }).reset_index()

    # 全ハンドリングプロパティのダミーデータフレームを作成
    all_properties_df = pd.DataFrame({'handling_property_id': handling_property_df['handling_property_id'].unique()})
    all_properties_df['key'] = 0

    # 元のデータフレームにキーを追加して全ハンドリングプロパティと結合
    user_handling_df['key'] = 0
    merged_df = user_handling_df.merge(all_properties_df, on='key', how='left').drop('key', axis=1)

    # ハンドリングプロパティの存在をチェックしてダミー変数を設定
    merged_df['dummy'] = merged_df.apply(lambda row: 1 if row['handling_property_id_y'] in row['handling_property_id_x'] else 0, axis=1)

    # ダミー変数を元のデータフレームに結合
    dummy_df = merged_df.pivot(index='user_id', columns='handling_property_id_y', values='dummy').fillna(0).astype(int)
    dummy_df.columns = ['handling_property' + str(int(col)) for col in dummy_df.columns]

    user_handling_df = pd.concat([user_handling_df.set_index('user_id'), dummy_df], axis=1).reset_index()
    user_handling_df = user_handling_df.drop(columns=["handling_property_id"])

    # 性別を数値に変換
    user_handling_df['sex'] = user_handling_df['sex'].replace({'男': 0, '女': 1})

    # NaNを0に置き換え（欠損値処理）
    user_handling_df = user_handling_df.fillna(0)

    # user_idが空白の行を削除
    user_handling_df = user_handling_df[user_handling_df['user_id'] != 0]

    # インデックスをリセット（オプション）
    user_handling_df = user_handling_df.reset_index(drop=True)

    return user_handling_df

#userとcontentの紐づけ
def process_user_activity(user_handling_df, activity_df):
    user_cookie_df = user_handling_df[["user_id", "cookie_id"]]
    users_activity_df = user_cookie_df.merge(activity_df, on="cookie_id", how="right")
    users_activity_df = users_activity_df.dropna(subset=['user_id'])
    users_activity_df = users_activity_df.sort_values(by='user_id', ascending=True)
    print(users_activity_df)
    return users_activity_df

# 関数を使用してデータフレームを前処理
content_tag_df = process_content_data(content_df, tag_content_df, tags_df)
user_handling_df = process_user_data(users_df, user_property_df, handling_property_df)
users_activity_df = process_user_activity(user_handling_df, activity_df)

# Create a mapping from unique user indices to range [0, num_user_nodes):
unique_user_id = user_handling_df['user_id'].unique()
unique_user_id_df = pd.DataFrame(data={
    'user_id': unique_user_id,
    'mappedId': pd.RangeIndex(len(unique_user_id)),
})
print("Mapping of user IDs to consecutive values:")
print("==========================================")
print(unique_user_id_df.head())
print()

# Create a mapping from unique content indices to range [0, num_content_nodes):
unique_content_id = content_tag_df['content_id'].unique()
unique_content_id_df = pd.DataFrame(data={
    'content_id': unique_content_id,
    'mappedId': pd.RangeIndex(len(unique_content_id)),
})
print("Mapping of content IDs to consecutive values:")
print("===========================================")
print(unique_content_id_df.head())


# categorical カラムに
tag_columns = [col for col in content_tag_df.columns if 'tag' in col]
df_tag = content_tag_df[tag_columns]
content_feat = torch.from_numpy(df_tag.values).to(torch.float)
content_features_tensor = torch.tensor(df_tag.values, dtype=torch.float)

print("content_features_tensor",content_features_tensor)

user_feat = user_handling_df[['user_id','age', 'sex', 'industry', 'handling_property1',
    'handling_property2', 'handling_property3', 'handling_property4',
    'handling_property5', 'handling_property6', 'handling_property7',
    'handling_property8', 'handling_property9']
    ]
user_feat_dummy = pd.get_dummies(user_feat["industry"])
# ダミー変数のデータフレームを整数型に変換
user_feat_dummy = user_feat_dummy.astype(int)
user_feat = pd.concat([user_feat, user_feat_dummy], axis=1)
user_feat = user_feat.drop(columns=["industry"])
user_feat_tensor = user_feat.drop(columns=["user_id"])
user_feat_tensor = torch.from_numpy(user_feat_tensor.values).to(torch.float)


# ID の生成
activity_user_id = pd.merge(users_activity_df['user_id'], unique_user_id_df, on='user_id', how='left')
activity_user_id = torch.from_numpy(activity_user_id['mappedId'].values)

activity_content_id = pd.merge(users_activity_df["content_id"], unique_content_id_df,
                            left_on='content_id', right_on='content_id', how='left')
activity_content_id = torch.from_numpy(activity_content_id['mappedId'].values)

# edge を設定
edge_index_user_to_content = torch.stack([activity_user_id, activity_content_id], dim=0)

data = HeteroData()

# ノードの割当
data['user'].node_id = torch.arange(len(unique_user_id_df))
data['content'].node_id = torch.arange(len(unique_content_id_df))
#movieの削除
# del data['movie']


# 属性の割当
data["content"].x = content_feat
data["user"].x = user_feat_tensor
data["user", "activity", "content"].edge_index = edge_index_user_to_content

# 無向グラフにする。
data = T.ToUndirected()(data)
print(data)

#user_id毎のedge_label_indexの生成
def create_edge_label_index(user_id):
    #このセルで予測するエッジを設定。
    N=len(unique_content_id)

    # ユーザーIDが1であることを示すテンソルを作成
    user_ids = torch.full((N,), user_id-1, dtype=torch.long)  # 全ての要素がユーザーID 1のテンソル

    # コンテンツIDを示すテンソルを作成
    content_ids = torch.arange(N)  # 0からN-1までのコンテンツID

    # edge_label_indexを作成
    edge_all_label_index = torch.stack([user_ids, content_ids], dim=0)

    return edge_all_label_index

#dataの設定
def user_data(user_id, user_feat, new_data, content_features_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    user_feat = user_feat.loc[user_feat['user_id'] == user_id].drop(['user_id'], axis=1)
    user_feat_tensor = torch.from_numpy(user_feat.values).to(torch.float)

    # ノードの割当
    new_data['user'].node_id = torch.arange(len(unique_user_id))
    new_data['content'].node_id = torch.arange(len(content_df))

    # 属性の割当
    new_data["content"].x = content_features_tensor
    new_data["user"].x = user_feat_tensor
    # new_data["user", "activity", "content"].edge_label_index = edge_index_specific_user_to_content
    new_data["user", "activity", "content"].edge_index = edge_index_user_to_content

    # ここでnew_dataオブジェクト全体をデバイスに移動させます
    new_data = new_data.to(device)

    #  # 正のエッジサンプル (実際に存在するエッジ)
    # edge_label_index = new_data['user', 'activity', 'content'].edge_index

    # 無向グラフにする。
    new_data = T.ToUndirected()(new_data)
    new_data = new_data.to(device)

    return new_data

#特定のuser_idでトップｎの確率を返すようにする。
def recommend_user(user_id,prediction,top_k=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 除外したい特定のコンテンツIDのリスト
    # 特定のユーザーに関連するアクティビティをフィルタリング
    filtered_activity_df = users_activity_df[users_activity_df['user_id'] == user_id]
    checked_content = filtered_activity_df["content_id"].tolist()

    node_ids = torch.arange(len(content_df), device=device)  # コンテンツIDのテンソル

    # 除外リストに含まれないコンテンツIDのみを保持するためのマスクを作成
    mask = torch.ones_like(prediction, dtype=torch.bool, device=device)  # 最初に全てを含むマスクを作成
    for exclude_id in checked_content:
        mask &= (node_ids != exclude_id)

    # predictionをデバイスに移動
    prediction = prediction.to(device)

    filtered_predictions = prediction[mask]
    filtered_node_ids = node_ids[mask]

    # フィルタリングされた確率に基づいてテンソルを降順にソート
    sorted_predictions, indices = torch.sort(filtered_predictions, descending=True)

    # 上位3つのノードIDを表示
    for node_id, prob in zip(filtered_node_ids[indices][:3], sorted_predictions[:3]):
        print(f'Node ID: {node_id.item()}, Probability: {prob.item():.2f}')

    # 上位3つのノードIDを取得（存在する場合）
    top_n = 3  # 取得したい上位の数
    top_node_ids = filtered_node_ids[indices][:top_n].tolist()

    # user_id=1に対して、上位3つのノードIDを含む辞書を作成
    result_dict = {'user_id': user_id,'recommend_content':top_node_ids}

    return result_dict

# モデルの読み込み（必要な場合）

class GNN(torch.nn.Module):
    def __init__(self, hidden_sizes,dropout_rate):
        super().__init__()

        self.conv1 = SAGEConv(hidden_sizes, hidden_sizes)
        self.conv2 = SAGEConv(hidden_sizes, hidden_sizes)

        self.dropout = Dropout(p=dropout_rate)  # ドロップアウト:(p=ドロップアウト率)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        x = self.dropout(x)
        return x

class Classifier(torch.nn.Module):
    def forward(self, x_user, x_content, edge_label_index):
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_content = x_content[edge_label_index[1]]

        return (edge_feat_user * edge_feat_content).sum(dim=-1)

class Model(torch.nn.Module):
    def __init__(self, hidden_sizes, dropout_rate, new_data,num_user_features, num_content_features, num_users, num_contents):
        super().__init__()

        self.content_lin = Linear(num_content_features, hidden_sizes)
        self.user_lin = Linear(num_user_features, hidden_sizes)
        self.user_emb = Embedding(num_users, hidden_sizes)
        self.content_emb = Embedding(num_contents, hidden_sizes)

        self.gnn = GNN(hidden_sizes, dropout_rate)
        self.gnn = to_hetero(self.gnn, metadata=new_data.metadata())
        self.classifier = Classifier()

    def forward(self, new_data):

        # user_data = new_data['user'].x
        # content_data = new_data['content'].x

        # ユーザーノードの特徴行列からすべてのインデックスを取得
        user_indices = torch.arange(data['user'].x.size(0)).to(device)

        # コンテンツノードの特徴行列からすべてのインデックスを取得
        content_indices = torch.arange(data['content'].x.size(0)).to(device)

        x_dict = {
            'user': self.user_lin(new_data['user']['x']) + self.user_emb(user_indices),
            'content': self.content_lin(new_data['content']['x']) + self.content_emb(content_indices),
        }
        x_dict = self.gnn(x_dict, new_data.edge_index_dict)
        pred = self.classifier(
            x_dict['user'],
            x_dict['content'],
            new_data['user', 'activity', 'content'].edge_label_index,
        )
        return pred

@app.get("/")
def index():
    return "Hello world"

@app.get("/user_id")
def gnn_predict(user_id:int):
    new_data = HeteroData()

    hidden_sizes=64
    dropout_rate=0.0
    num_user_features=17
    num_content_features=398
    num_users=3000
    num_contents=330

    result = []

    new_data = user_data(user_id,user_feat,new_data,content_features_tensor)
    new_data["user", "activity", "content"].edge_label_index = create_edge_label_index(user_id)

    #モデルの読み込み
    model = Model(hidden_sizes, dropout_rate, new_data, num_user_features, num_content_features, num_users, num_contents)  # モデルのインスタンスを作成
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.to(device)

    result = []

    print(user_id,
            # create_edge_label_index(target_id)
            )

    model.eval()  # モデルを評価モードに設定


    # 予測実行
    with torch.no_grad():
        prediction = model(new_data)
        prediction = torch.sigmoid(prediction)  # ロジットを確率に変換
        result.append(recommend_user(user_id,prediction,3))

    return result