import torch
import numpy as np
from datetime import datetime, UTC

class EngagementPredictor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._construct_emb_layers()
        self.tweet_layers = torch.nn.Sequential(
            torch.nn.Linear(1024,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Linear(128,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
        )
        self.user_layers = torch.nn.Sequential(
            torch.nn.Linear(1024,128),
            torch.nn.BatchNorm1d(128),
            torch.nn.GELU(),
            torch.nn.Linear(128,64),
            torch.nn.BatchNorm1d(64),
            torch.nn.GELU(),
        )
        self.output_layer = torch.nn.Linear(128,1)
        # self.dropout = torch.nn.Dropout(0.8)

    def _construct_emb_layers(self):
        self.embs =  {
            'verified': torch.nn.Embedding(2,128),
            'months': torch.nn.Embedding(300,256),
            'following': torch.nn.Embedding(100,256),
            'followers': torch.nn.Embedding(100,256),
            'user_type': torch.nn.Embedding(2,128),
        }

    def parse_input(self,input_dicts):
        text_emb = torch.FloatTensor([doc["_source"].get("embedding",[0]*1024) for doc in input_dicts])
        user_emb = self.get_user_emb([doc['_source']['twitter_tweet_author_detail'] for doc in input_dicts])
        return text_emb, user_emb

    def user_feature_process(self,user_dict):
        verified = 1 if user_dict['twitter_user_verified'] else 0
        months = int((datetime.now(UTC) - datetime.fromisoformat(user_dict['twitter_user_created_at'])).days/30)
        following = int(user_dict['twitter_user_following_count'])
        # print(following)
        followers = int(user_dict['twitter_user_followers_count'])
        user_type = user_dict['twitter_user_tags_v2'].get('user_type','Individual')
        user_type = 1 if user_type == 'Individual' else 0
        months = 300 if months>300 else months
        following = int(np.log2(following)) if following>0 else 0
        followers = int(np.log2(followers)) if followers>0 else 0
        return {
            'verified': verified,
            'months': months,
            'following': following,
            'followers': followers,
            'user_type': user_type
        }

    def get_user_emb(self, user_dicts):
        emb_list = []
        for key, emb_layer in self.embs.items():
            # print(key)
            emb_list.append(
                self.embs[key](torch.IntTensor([self.user_feature_process(item)[key] for item in user_dicts]))
            )
        user_emb = torch.cat(emb_list,axis=1)
        return user_emb
    
    def forward(self,x):
        x_tweet, x_user = self.parse_input(x)
        tweet_feat = self.tweet_layers(x_tweet)
        user_feat = self.user_layers(x_user)
        output = self.output_layer(torch.cat([tweet_feat,user_feat],axis=-1))
        return torch.nn.functional.sigmoid(output)