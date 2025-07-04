import torch
import numpy as np
import pandas as pd
import ijson
import json
import os
from pathlib import Path
from transformers import pipeline
from datetime import datetime as dt
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

root_path = os.getcwd() + "/Dataset"


class Twibot20(Dataset):
    """
    Twibot-20 Dataset based on the BotGNN models build by Yung et al. 2020
    """

    def __init__(self, root=root_path, device="cpu", process=True, save=True):
        """
        Create a Twibot-20 dataset.
        """
        self.root = root
        self.device = device
        self.process = process
        if process:
            print(f"Root path {root_path}")
            print("Loading train.json")
            df_train = pd.read_json(root_path + "/Twibot-20/train.json")
            print("Loading test.json")
            df_test = pd.read_json(root_path + "/Twibot-20/test.json")
            print("Loading support.json")
            support_data = []
            with open(root_path + "/Twibot-20/support.json", "r") as f:
                objects = ijson.items(
                    f, "item"
                )  # 'item' refers to each item in the outermost list
                for i, obj in enumerate(objects):
                    support_data.append(obj)
                    if i % 50000 == 0:  # Every 50K objects
                        print(f"Loaded {i} records...")
            df_support = pd.DataFrame(
                support_data
            )  # Loading using this method to avoid overloading memory on smaller system
            print("Loading dev.json")
            df_dev = pd.read_json(root_path + "/Twibot-20/dev.json")
            print("Finished")
            df_train = df_train.iloc[:, [0, 1, 2, 3, 5]]
            df_test = df_test.iloc[:, [0, 1, 2, 3, 5]]
            df_support = df_support.iloc[:, [0, 1, 2, 3]]
            df_dev = df_dev.iloc[:, [0, 1, 2, 3, 5]]
            df_support["label"] = "None"
            self.df_data_labeled = pd.concat(
                [df_train, df_dev, df_test], ignore_index=True
            )
            self.df_data = pd.concat(
                [df_train, df_dev, df_test, df_support], ignore_index=True
            )
            self.df_data = self.df_data
            self.df_data_labeled = self.df_data_labeled
        self.save = save

    def load_labels(self):
        print("Loading labels...", end="   ")
        path = self.root + "/Twibot-20/processed_data/label.pt"
        if not os.path.exists(path):
            labels = torch.LongTensor(self.df_data_labeled["label"]).to(self.device)
            if self.save:
                torch.save(labels, root_path + "/Twibot-20/proccessed_data/label.pt")
        else:
            labels = torch.load(self.root + "/Twibot-20/processed_data/label.pt").to(
                self.device
            )
        print("Finished")

        return labels

    def Des_Preprocess(self):
        print("Loading raw feature1...", end="   ")
        path = self.root + "/Twibot-20/description.npy"
        if not os.path.exists(path):
            description = []
            for i in range(self.df_data.shape[0]):
                if (
                    self.df_data["profile"][i] is None
                    or self.df_data["profile"][i]["description"] is None
                ):
                    description.append("None")
                else:
                    description.append(self.df_data["profile"][i]["description"])
            description = np.array(description)
            if self.save:
                np.save(path, description)
        else:
            description = np.load(path, allow_pickle=True)
        print("Finished")
        return description

    def Des_embbeding(self):
        print("Running feature1 embedding")
        path = self.root + "/Twibot-20/processed_data/des_tensor.pt"
        if not os.path.exists(path):
            description = np.load(
                self.root + "/Twibot-20/description.npy", allow_pickle=True
            )
            print("Loading RoBerta")
            feature_extraction = pipeline(
                "feature-extraction",
                model="distilroberta-base",
                tokenizer="distilroberta-base",
                device=0,
            )
            des_vec = []
            # for (j,each) in tqdm(enumerate(description)):
            for each in tqdm(description):
                feature = torch.Tensor(feature_extraction(each))
                for i, tensor in enumerate(feature[0]):
                    if i == 0:
                        feature_tensor = tensor
                    else:
                        feature_tensor += tensor
                feature_tensor /= feature.shape[1]
                des_vec.append(feature_tensor)
                # if (j%1000==0):
                # print('[{:>6d}/229580]'.format(j+1))
            des_tensor = torch.stack(des_vec, 0).to(self.device)
            if self.save:
                torch.save(
                    des_tensor, root_path + "/Twibot-20/processed_data/des_tensor.pt"
                )
        else:
            des_tensor = torch.load(
                self.root + "/Twibot-20/processed_data/des_tensor.pt"
            ).to(self.device)
        print("Finished")
        return des_tensor

    def tweets_preprocess(self):
        print("Loading raw feature2...", end="   ")
        path = self.root + "/Twibot-20/processed_data/tweets.npy"
        if not os.path.exists(path):
            tweets = []
            for i in range(self.df_data.shape[0]):
                one_usr_tweets = []
                if self.df_data["tweet"][i] is None:
                    one_usr_tweets.append("")
                else:
                    for each in self.df_data["tweet"][i]:
                        one_usr_tweets.append(each)
                tweets.append(one_usr_tweets)
            tweets = np.array(tweets)
            if self.save:
                np.save(path, tweets)
        else:
            tweets = np.load(path, allow_pickle=True)
        print("Finished")
        return tweets

    def tweets_embedding(self):
        print("Running feature2 embedding")
        path = self.root + "/Twibot-20/processed_data/tweets_tensor.pt"
        if not os.path.exists(path):
            tweets = np.load("tweets.npy", allow_pickle=True)
            print("Loading RoBerta")
            feature_extract = pipeline(
                "feature-extraction",
                model="roberta-base",
                tokenizer="roberta-base",
                device=0,
                padding=True,
                truncation=True,
                max_length=500,
                add_special_tokens=True,
            )
            tweets_list = []
            for each_person_tweets in tqdm(tweets):
                for j, each_tweet in enumerate(each_person_tweets):
                    each_tweet_tensor = torch.tensor(feature_extract(each_tweet))
                    for k, each_word_tensor in enumerate(each_tweet_tensor[0]):
                        if k == 0:
                            total_word_tensor = each_word_tensor
                        else:
                            total_word_tensor += each_word_tensor
                    total_word_tensor /= each_tweet_tensor.shape[1]
                    if j == 0:
                        total_each_person_tweets = total_word_tensor
                    else:
                        total_each_person_tweets += total_word_tensor
                total_each_person_tweets /= len(each_person_tweets)
                tweets_list.append(total_each_person_tweets)
                # if (i%500==0):
                # print('[{:>6d}/229580]'.format(i+1))
            tweet_tensor = torch.stack(tweets_list).to(self.device)
            if self.save:
                torch.save(tweet_tensor, path)
        else:
            tweets_tensor = torch.load(
                self.root + "/Twibot-20/processed_data/tweets_tensor.pt"
            ).to(self.device)
        print("Finished")
        return tweets_tensor

    def num_prop_preprocess(self):
        print("Processing feature3...", end="   ")
        path0 = self.root + "/Twibot-20/processed_data/num_properties_tensor.pt"
        if not os.path.exists(path0):
            path = self.root
            if not os.path.exists(path + "followers_count.pt"):
                followers_count = []
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["followers_count"] is None
                    ):
                        followers_count.append(0)
                    else:
                        followers_count.append(
                            self.df_data["profile"][i]["followers_count"]
                        )
                followers_count = torch.tensor(
                    np.array(followers_count, dtype=np.float32)
                ).to(self.device)
                if self.save:
                    torch.save(followers_count, path + "followers_count.pt")

                friends_count = []
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["friends_count"] is None
                    ):
                        friends_count.append(0)
                    else:
                        friends_count.append(
                            self.df_data["profile"][i]["friends_count"]
                        )
                friends_count = torch.tensor(
                    np.array(friends_count, dtype=np.float32)
                ).to(self.device)
                if self.save:
                    torch.save(friends_count, path + "friends_count.pt")

                screen_name_length = []
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["screen_name"] is None
                    ):
                        screen_name_length.append(0)
                    else:
                        screen_name_length.append(
                            len(self.df_data["profile"][i]["screen_name"])
                        )
                screen_name_length = torch.tensor(
                    np.array(screen_name_length, dtype=np.float32)
                ).to(self.device)
                if self.save:
                    torch.save(screen_name_length, path + "screen_name_length.pt")

                favourites_count = []
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["favourites_count"] is None
                    ):
                        favourites_count.append(0)
                    else:
                        favourites_count.append(
                            self.df_data["profile"][i]["favourites_count"]
                        )
                favourites_count = torch.tensor(
                    np.array(favourites_count, dtype=np.float32)
                ).to(self.device)
                if self.save:
                    torch.save(favourites_count, path + "favourites_count.pt")

                active_days = []
                date0 = dt.strptime(
                    "Tue Sep 1 00:00:00 +0000 2020 ", "%a %b %d %X %z %Y "
                )
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["created_at"] is None
                    ):
                        active_days.append(0)
                    else:
                        date = dt.strptime(
                            self.df_data["profile"][i]["created_at"],
                            "%a %b %d %X %z %Y ",
                        )
                        active_days.append((date0 - date).days)
                active_days = torch.tensor(np.array(active_days, dtype=np.float32)).to(
                    self.device
                )
                if self.save:
                    torch.save(active_days, path + "active_days.pt")

                statuses_count = []
                for i in range(self.df_data.shape[0]):
                    if (
                        self.df_data["profile"][i] is None
                        or self.df_data["profile"][i]["statuses_count"] is None
                    ):
                        statuses_count.append(0)
                    else:
                        statuses_count.append(
                            int(self.df_data["profile"][i]["statuses_count"])
                        )
                statuses_count = torch.tensor(
                    np.array(statuses_count, dtype=np.float32)
                ).to(self.device)
                if self.save:
                    torch.save(statuses_count, path + "statuses_count.pt")

            else:
                active_days = torch.load(path + "active_days.pt")
                screen_name_length = torch.load(path + "screen_name_length.pt")
                favourites_count = torch.load(path + "favourites_count.pt")
                followers_count = torch.load(path + "followers_count.pt")
                friends_count = torch.load(path + "friends_count.pt")
                statuses_count = torch.load(path + "statuses_count.pt")

            active_days = pd.Series(active_days.to("cpu").detach().numpy())
            active_days = (active_days - active_days.mean()) / active_days.std()
            active_days = torch.tensor(np.array(active_days))

            screen_name_length = pd.Series(
                screen_name_length.to("cpu").detach().numpy()
            )
            screen_name_length_days = (
                screen_name_length - screen_name_length.mean()
            ) / screen_name_length.std()
            screen_name_length_days = torch.tensor(np.array(screen_name_length_days))

            favourites_count = pd.Series(favourites_count.to("cpu").detach().numpy())
            favourites_count = (
                favourites_count - favourites_count.mean()
            ) / favourites_count.std()
            favourites_count = torch.tensor(np.array(favourites_count))

            followers_count = pd.Series(followers_count.to("cpu").detach().numpy())
            followers_count = (
                followers_count - followers_count.mean()
            ) / followers_count.std()
            followers_count = torch.tensor(np.array(followers_count))

            friends_count = pd.Series(friends_count.to("cpu").detach().numpy())
            friends_count = (friends_count - friends_count.mean()) / friends_count.std()
            friends_count = torch.tensor(np.array(friends_count))

            statuses_count = pd.Series(statuses_count.to("cpu").detach().numpy())
            statuses_count = (
                statuses_count - statuses_count.mean()
            ) / statuses_count.std()
            statuses_count = torch.tensor(np.array(statuses_count))

            num_prop = torch.cat(
                (
                    followers_count.reshape([229580, 1]),
                    friends_count.reshape([229580, 1]),
                    favourites_count.reshape([229580, 1]),
                    statuses_count.reshape([229580, 1]),
                    screen_name_length_days.reshape([229580, 1]),
                    active_days.reshape([229580, 1]),
                ),
                1,
            ).to(self.device)

            if self.save:
                torch.save(
                    num_prop,
                    self.root + "/Twibot-20/processed_data/num_properties_tensor.pt",
                )

        else:
            num_prop = torch.load(
                self.root + "/Twibot-20/processed_data/num_properties_tensor.pt"
            ).to(self.device)
        print("Finished")
        return num_prop

    def cat_prop_preprocess(self):
        print("Processing feature4...", end="   ")
        path = self.root + "/Twibot-20/processed_data/cat_properties_tensor.pt"
        if not os.path.exists(path):
            category_properties = []
            properties = [
                "protected",
                "geo_enabled",
                "verified",
                "contributors_enabled",
                "is_translator",
                "is_translation_enabled",
                "profile_background_tile",
                "profile_use_background_image",
                "has_extended_profile",
                "default_profile",
                "default_profile_image",
            ]
            for i in range(self.df_data.shape[0]):
                prop = []
                if self.df_data["profile"][i] is None:
                    for i in range(11):
                        prop.append(0)
                else:
                    for each in properties:
                        if self.df_data["profile"][i][each] is None:
                            prop.append(0)
                        else:
                            if self.df_data["profile"][i][each] == "True ":
                                prop.append(1)
                            else:
                                prop.append(0)
                prop = np.array(prop)
                category_properties.append(prop)
            category_properties = torch.tensor(
                np.array(category_properties, dtype=np.float32)
            ).to(self.device)
            if self.save:
                torch.save(
                    category_properties,
                    self.root + "/Twibot-20/processed_data/cat_properties_tensor.pt",
                )
        else:
            category_properties = torch.load(
                self.root + "/Twibot-20/processed_data/cat_properties_tensor.pt"
            ).to(self.device)
        print("Finished")
        return category_properties

    def Build_Graph(self):
        print("Building graph", end="   ")
        path = self.root + "/Twibot-20/processed_data/edge_index.pt"
        if not os.path.exists(path):
            id2index_dict = {id: index for index, id in enumerate(self.df_data["ID"])}
            edge_index = []
            edge_type = []
            for i, relation in enumerate(self.df_data["neighbor"]):
                if relation is not None:
                    for each_id in relation["following"]:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                        edge_type.append(0)
                    for each_id in relation["follower"]:
                        try:
                            target_id = id2index_dict[int(each_id)]
                        except KeyError:
                            continue
                        else:
                            edge_index.append([i, target_id])
                        edge_type.append(1)
                else:
                    continue
            edge_index = (
                torch.tensor(edge_index, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device)
            )
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)
            if self.save:
                torch.save(
                    edge_index, self.root + "/Twibot-20/processed_data/edge_index.pt"
                )
                torch.save(
                    edge_type, self.root + "/Twibot-20/processed_data/edge_type.pt"
                )
        else:
            edge_index = torch.load(
                self.root + "/Twibot-20/processed_data/edge_index.pt"
            ).to(self.device)
            edge_type = torch.load(
                self.root + "/Twibot-20/processed_data/edge_type.pt"
            ).to(self.device)
            print("Finished")
        return edge_index, edge_type

    def Build_Feature_Graph(
        self,
        des_tensor,
        tweets_tensor,
        num_prop,
        category_prop,
        k=5,
        name="feature_graph",
    ):
        """
        Memory-efficient cosine similarity graph using top-k nearest neighbors.neighbours
        """
        print("Building feature-based graph from all features (chunked)...", end="   ")
        save_dir = os.path.join(self.root, "processed_data")
        edge_path = os.path.join(save_dir, f"{name}_edge_index.pt")
        type_path = os.path.join(save_dir, f"{name}_edge_type.pt")

        if not os.path.exists(edge_path) or not os.path.exists(type_path):
            # Concatenate all features
            features = torch.cat(
                [des_tensor, tweets_tensor, num_prop, category_prop], dim=1
            )  # [N, D]
            features = torch.nn.functional.normalize(
                features, dim=1
            )  # L2 normalize row-wise
            n_nodes = features.size(0)

            edge_index = []
            edge_type = []

            batch_size = 1024  # Tune this based on memory
            for start in range(0, n_nodes, batch_size):
                end = min(start + batch_size, n_nodes)
                batch = features[start:end]  # [B, D]
                sims = torch.matmul(batch, features.T)  # [B, N]
                sims[:, start:end] = -1  # mask self-comparisons

                topk_values, topk_indices = torch.topk(sims, k=k, dim=1)

                for i in range(end - start):
                    src = start + i
                    neighbors = topk_indices[i]
                    for dst in neighbors:
                        edge_index.append([src, dst.item()])
                        edge_type.append(0)

            edge_index = (
                torch.tensor(edge_index, dtype=torch.long)
                .t()
                .contiguous()
                .to(self.device)
            )
            edge_type = torch.tensor(edge_type, dtype=torch.long).to(self.device)

            if self.save:
                os.makedirs(save_dir, exist_ok=True)
                torch.save(edge_index, edge_path)
                torch.save(edge_type, type_path)
        else:
            edge_index = torch.load(edge_path).to(self.device)
            edge_type = torch.load(type_path).to(self.device)
            print("Loaded from cache")

        print("Finished")
        return edge_index, edge_type

    def train_val_test_mask(self):
        train_idx = range(8278)
        val_idx = range(8278, 8278 + 2365)
        test_idx = range(8278 + 2365, 8278 + 2365 + 1183)
        return train_idx, val_idx, test_idx

    def dataloader(self, build_feature_graph=False):
        labels = self.load_labels()
        if self.process:
            self.Des_Preprocess()
            self.tweets_preprocess()
        des_tensor = self.Des_embbeding()
        tweets_tensor = self.tweets_embedding()
        num_prop = self.num_prop_preprocess()
        category_prop = self.cat_prop_preprocess()
        if build_feature_graph:
            edge_index, edge_type = self.Build_Feature_Graph(
                des_tensor, tweets_tensor, num_prop, category_prop
            )  # Build graph based on feature instead of relationship.
        else:
            edge_index, edge_type = self.Build_Graph()
        train_idx, val_idx, test_idx = self.train_val_test_mask()
        return (
            des_tensor,
            tweets_tensor,
            num_prop,
            category_prop,
            edge_index,
            edge_type,
            labels,
            train_idx,
            val_idx,
            test_idx,
        )


class Instafake(Dataset):
    """
    Data loader for Instafake dataset based on the repo: https://github.com/fcakyon/instafake-dataset/tree/master/data
    """

    def __init__(self, root, dataset_version, device="cpu", process=True):
        self.root = Path(root).parents / "Dataset/InstaFake/interim"
        self.device = device
        self.df = pd.read_csv(self.root / "combined_account_data.csv", engine="pyarrow")

        # Preprocess dataframe (e.g., normalize, fillna)
        if process:
            self._preprocess()

        # Prepare tensors
        self.features, self.labels = self._prepare_tensors()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def _preprocess(self):
        # Example: fill NaNs with 0
        self.dataframe = self.dataframe.fillna(0)

    def _prepare_tensors(self):
        df = self.dataframe

        if self.dataset_type == "automated":
            label_col = "automated_behaviour"
        elif self.dataset_type == "fake":
            label_col = "is_fake"
        else:
            raise ValueError("Unknown dataset type")

        features = df.drop(columns=[label_col]).astype(float).values
        labels = df[label_col].astype(int).values

        features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long).to(self.device)

        return features_tensor, labels_tensor

    def _create_dataframe(self, account_data_list, dataset_type):
        rows = []
        for account_data in account_data_list:
            user_follower_count = account_data["userFollowerCount"]
            user_following_count = account_data["userFollowingCount"]
            follower_following_ratio = user_follower_count / max(
                1, user_following_count
            )

            if dataset_type == "automated":
                row = {
                    "user_media_count": account_data["userMediaCount"],
                    "user_follower_count": user_follower_count,
                    "user_following_count": user_following_count,
                    "user_has_highligh_reels": account_data["userHasHighlighReels"],
                    "user_has_external_url": account_data["userHasExternalUrl"],
                    "user_tags_count": account_data["userTagsCount"],
                    "follower_following_ratio": follower_following_ratio,
                    "user_biography_length": account_data["userBiographyLength"],
                    "username_length": account_data["usernameLength"],
                    "username_digit_count": account_data["usernameDigitCount"],
                    "media_comment_numbers": account_data["mediaCommentNumbers"],
                    "media_comments_are_disabled": account_data["mediaCommentNumbers"],
                    "media_has_location_info": account_data["mediaHasLocationInfo"],
                    "media_hashtag_numbers": account_data["mediaHashtagNumbers"],
                    "media_like_numbers": account_data["mediaLikeNumbers"],
                    "mediaUpload_times": account_data["mediaUploadTimes"],
                    "automated_behaviour": account_data["automatedBehaviour"],
                }
            elif dataset_type == "fake":
                row = {
                    "user_media_count": account_data["userMediaCount"],
                    "user_follower_count": user_follower_count,
                    "user_following_count": user_following_count,
                    "user_has_profil_pic": account_data["userHasProfilPic"],
                    "user_is_private": account_data["userIsPrivate"],
                    "follower_following_ratio": follower_following_ratio,
                    "user_biography_length": account_data["userBiographyLength"],
                    "username_length": account_data["usernameLength"],
                    "username_digit_count": account_data["usernameDigitCount"],
                    "is_fake": account_data["isFake"],
                }
            rows.append(row)

        return pd.DataFrame(rows)

    def _import_data(self, dataset_path, dataset_version):
        dataset_type = re.findall("automated|fake", dataset_version)[0]

        if dataset_type == "automated":
            with open(
                os.path.join(dataset_path, dataset_version, "automatedAccountData.json")
            ) as f:
                automated_data = json.load(f)
            with open(
                os.path.join(
                    dataset_path, dataset_version, "nonautomatedAccountData.json"
                )
            ) as f:
                nonautomated_data = json.load(f)

            df_automated = self._create_dataframe(automated_data, dataset_type)
            df_nonautomated = self._create_dataframe(nonautomated_data, dataset_type)
            df_merged = pd.concat([df_automated, df_nonautomated], ignore_index=True)

        elif dataset_type == "fake":
            with open(
                os.path.join(dataset_path, dataset_version, "fakeAccountData.json")
            ) as f:
                fake_data = json.load(f)
            with open(
                os.path.join(dataset_path, dataset_version, "realAccountData.json")
            ) as f:
                real_data = json.load(f)

            df_fake = self._create_dataframe(fake_data, dataset_type)
            df_real = self._create_dataframe(real_data, dataset_type)
            df_merged = pd.concat([df_fake, df_real], ignore_index=True)

        return {"dataset_type": dataset_type, "dataframe": df_merged}
