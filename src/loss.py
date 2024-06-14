import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform
from scipy.stats import pearsonr

def correlation_coeff(values, z):
    """
    Compute the correlation coefficient between the values and the latent variables.
    
    Parameters:
    - values (torch.Tensor): Values of the functions.
    - z (torch.Tensor): Latent variables.
    
    Returns:
    - torch.Tensor: Correlation coefficient between the values and the latent variables.
    """

    distance_values = torch.nan_to_num(torch.cdist(values, values, p=1.0)/25, posinf=1000, neginf=-1000)

    area_values = []     
    #for i,_ in enumerate(values):
    #    for k,_ in enumerate(values):
    #        area_values.append(torch.nan_to_num(torch.sum(torch.abs(torch.sub(values[i], values[k]))))/2)

    correlation_values = torch.nan_to_num(torch.corrcoef(values), posinf=1000, neginf=-1000)
    distance_z = torch.nan_to_num(torch.cdist(z, z, p=1.0), posinf=1000, neginf=-1000)

    correlation_dis = torch.corrcoef(torch.stack((distance_values.flatten(), distance_z.flatten())))[0, 1]
    correlation_cor = torch.corrcoef(torch.stack((correlation_values.flatten(), distance_z.flatten())))[0, 1]
    covariance_dis = torch.cov(torch.stack((distance_values.flatten(), distance_z.flatten())))[0, 1]
    covaraince_cor = torch.cov(torch.stack((correlation_values.flatten(), distance_z.flatten())))[0, 1]
    
    return correlation_cor, correlation_dis, covaraince_cor, covariance_dis


class LatentCorrelationLoss(nn.Module):
    def __init__(self):
        super(LatentCorrelationLoss, self).__init__()
    
    def forward(self, values, z):
        correlation_cor, correlation_dis, covariance_cor, covariance_dis = correlation_coeff(values, z)
        #return -covariance_dis
        #return 1/((covariance_dis+1)*0.5)
        return - correlation_dis + 1 
        #return correlation_dis + 1 #1/((correlation_cor+1)*0.5)
        #return -(correlation_cor + 1) +2


# NOT IMPLEMENTED IN THE FINAL VERSION
class LabelDifference(nn.Module):
    '''
    Computes the difference between labels (function names) using manhattan distance
    return: Difference matrix [bs, bs]
    '''
    def __init__(self, distance_type='l1'):
        super(LabelDifference, self).__init__()
        self.distance_type = distance_type

    def forward(self, labels):
        # labels: [bs, label_dim]
        # output: [bs, bs]
        if self.distance_type == 'l1':
            return torch.abs(labels[:, None, :] - labels[None, :, :]).sum(dim=-1)
        else:
            raise ValueError(self.distance_type)

# NOT IMPLEMENTED IN THE FINAL VERSION
class FeatureSimilarity(nn.Module):
    '''
    Compute similarity between function values using euclidean distance

    retrun: Similarity matrix [bs, bs]
    '''
    def __init__(self, similarity_type='l2'):
        super(FeatureSimilarity, self).__init__()
        self.similarity_type = similarity_type

    def forward(self, features):
        # labels: [bs, feat_dim]
        # output: [bs, bs]
        if self.similarity_type == 'l2':
            return jaccard_similarity_matrix(features)#- (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)
        else:
            raise ValueError(self.similarity_type)

# NOT IMPLEMENTED IN THE FINAL VERSION
class RnCLoss(nn.Module):
    def __init__(self, temperature=2, label_diff='l1', feature_sim='l2'):
        super(RnCLoss, self).__init__()
        self.t = temperature
        self.label_diff_fn = LabelDifference(label_diff)
        self.feature_sim_fn = FeatureSimilarity(feature_sim)

    def forward(self, features, labels):
        # features: [bs, 2, feat_dim]
        # labels: [bs, label_dim]

        features = torch.cat([features[:, 0], features[:, 1]], dim=0)  # [2bs, feat_dim]
        labels = labels.repeat(2, 1)  # [2bs, label_dim]

        label_diffs = self.label_diff_fn(labels)
        logits = self.feature_sim_fn(features).div(self.t)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits -= logits_max.detach()
        exp_logits = logits.exp()

        n = logits.shape[0]  # n = 2bs

        # remove diagonal
        logits = logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        exp_logits = exp_logits.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)
        label_diffs = label_diffs.masked_select((1 - torch.eye(n).to(logits.device)).bool()).view(n, n - 1)

        loss = 0.
        for k in range(n - 1):
            # for every function the value distance to the kth function
            pos_logits = logits[:, k]  # 2bs
            # for every function the jaccard similarity to the kth function
            pos_label_diffs = label_diffs[:, k]  # 2bs
            # bool matrix wether the corresponding element in is greater than or equal
            neg_mask = (label_diffs >= pos_label_diffs.view(-1, 1)).float()  # [2bs, 2bs - 1]
            # calculates the log-sum-exp operation
            pos_log_probs = pos_logits - torch.log((neg_mask * exp_logits).sum(dim=-1))  # 2bs
            loss += - (pos_log_probs / (n * (n - 1))).sum()
        return loss
    

# NOT IMPLEMENTED IN THE FINAL VERSION
def jaccard_similarity(tensor1, tensor2):
    """
    Compute Jaccard similarity between two binary tensors (NumPy arrays).
    
    Parameters:
    - tensor1, tensor2 (numpy.ndarray): Binary tensors (1 for presence, 0 for absence).
    
    Returns:
    - float: Jaccard similarity between the two tensors.
    """
    intersection_size = np.intersect1d(tensor1, tensor2).size
    union_size = np.union1d(tensor1, tensor2).size
    
    return - intersection_size / union_size if union_size != 0 else 0.0

# NOT IMPLEMENTED IN THE FINAL VERSION
def jaccard_similarity_matrix(tensor_list):
    """
    Compute the Jaccard similarity matrix for a list of binary tensors (NumPy arrays).
    
    Parameters:
    - tensor_list (list): A list of binary tensors (NumPy arrays) for which to compute Jaccard similarities.
    
    Returns:
    - numpy.ndarray: Jaccard similarity matrix.
    """
    n = len(tensor_list)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i, n):
            similarity = jaccard_similarity(tensor_list[i], tensor_list[j])
            similarity_matrix[i][j] = similarity
            similarity_matrix[j][i] = similarity  # The matrix is symmetric
    
    return torch.tensor(similarity_matrix)


