# -*- coding: utf-8 -*-
"""
@author: wem26
"""

num_of_epoch = 90
output_step = 500
dropout_rate_cnn = 0.2
dropout_rate = 0.2
celline = "IMR90"
script_id = celline+'_EPIANN'
out_dir = 'output/'+script_id
file_pre  = 'aug_50/'+celline


enhancer_length = 3000
promoter_length = 2000
BATCH_SIZE = 32

num_filters = [256] #1024, 256
inter_dim = [1] #2
e_conv_width = [15] #40, 40
pool_width = 30
dense_neuron = 32
dense_neuron_coor = [128, 64]
topk = 32
atten_hyper = 32
lamb = 10



############################################################################################################
import os
if os.path.exists(out_dir)==False:
    os.makedirs(out_dir)


import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops

import numpy as np
import pickle
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from Bio import SeqIO
import time


def one_hot_encode(sequences):
    sequence_length = len(sequences[0])
    integer_type = np.int8
    integer_array = LabelEncoder().fit(np.array(('ACGTN',)).view(integer_type)).transform(
        sequences.view(integer_type)).reshape(len(sequences), sequence_length)
    one_hot_encoding = OneHotEncoder(
        sparse=False, n_values=5, dtype=integer_type).fit_transform(integer_array)

    return one_hot_encoding.reshape(
        len(sequences), 1, sequence_length, 5).swapaxes(2, 3)[:, :, [0, 1, 2, 4], :]


def input_sequences(efile, pfile, labelfile):
                        seq_e = []
                        seq_p = []
                        enhancer_sequences = SeqIO.parse(open(efile), 'fasta')
                        promoter_sequences = SeqIO.parse(open(pfile), 'fasta')
                        for enhancer in enhancer_sequences:
                            name, sequence = enhancer.id, str(enhancer.seq)
                            seq_e.append(sequence)
    
                        for promoter in promoter_sequences:
                            name, sequence = promoter.id, str(promoter.seq)
                            seq_p.append(sequence)
    
                        seq_e = np.array(seq_e)
                        seq_p = np.array(seq_p)    
                        X_e = one_hot_encode(seq_e)
                        X_p = one_hot_encode(seq_p)
                        X_e = X_e.swapaxes(1,2)
                        X_e = X_e.swapaxes(2,3)
                        X_p = X_p.swapaxes(1,2)        
                        X_p = X_p.swapaxes(2,3)

    #split to seq_e_p, seq_e_n, seq_p_p and seq_p_n based on the labels
                        num_pos = 0
                        num_neg = 0    
                        with open(labelfile) as f:
                            for line in f:
                                if line.rstrip() == '1':
                                    num_pos = num_pos+1
                                else:
                                    num_neg = num_neg+1

                        return X_e[0:num_pos], X_e[num_pos:], X_p[0:num_pos], X_p[num_pos:]


def input_coordinates(efile, pfile, labelfile):
    X_e = np.loadtxt(efile)
    X_p = np.loadtxt(pfile)
    num_pos = 0
    num_neg = 0
    with open(labelfile) as f:
        for line in f:
            if line.rstrip() == '1':
                num_pos = num_pos+1
            else:
                num_neg = num_neg+1
    return X_e[0:num_pos], X_e[num_pos:], X_p[0:num_pos], X_p[num_pos:]


def input_stat(efile, pfile):
    X_e = np.loadtxt(efile)
    X_p = np.loadtxt(pfile)
    return X_e[0,0], X_e[0,1], X_e[1,0], X_e[1,1],X_p[0,0], X_p[0,1], X_p[1,0], X_p[1,1]


def get_coordinates_batches(seq_ep_coor, seq_en_coor,seq_pp_coor, seq_pn_coor, count, BATCH_SIZE):
    #based on the labels, stratified sampleing, positive part and negative part  
    #positive set    
    plen = len(seq_ep_coor)
    start_pos = count*BATCH_SIZE%plen
    end_pos = min((count+1)*BATCH_SIZE%plen, plen)
    
    if end_pos <= start_pos:
        ebatch_p = np.concatenate([seq_ep_coor[start_pos:plen], seq_ep_coor[0:end_pos]])
        pbatch_p = np.concatenate([seq_pp_coor[start_pos:plen], seq_pp_coor[0:end_pos]])
    else:
        ebatch_p = seq_ep_coor[start_pos: end_pos]
        pbatch_p = seq_pp_coor[start_pos: end_pos]
    
    
    #negative set
    nlen = len(seq_en_coor)
    start_pos = count*BATCH_SIZE%nlen
    end_pos = min((count+1)*BATCH_SIZE%nlen, nlen)
    
    if end_pos <= start_pos:
        ebatch_n = np.concatenate([seq_en_coor[start_pos:nlen], seq_en_coor[0:end_pos]])
        pbatch_n = np.concatenate([seq_pn_coor[start_pos:nlen], seq_pn_coor[0:end_pos]])
    else:
        ebatch_n = seq_en_coor[start_pos: end_pos]
        pbatch_n = seq_pn_coor[start_pos: end_pos]
    
    seq_e = np.concatenate([ebatch_p, ebatch_n])
    seq_p = np.concatenate([pbatch_p, pbatch_n])
    
    return seq_e, seq_p


def get_batches(seq_ep, seq_en,seq_pp, seq_pn, count, BATCH_SIZE):
    #based on the labels, stratified sampleing, positive part and negative part  
    #positive set    
    plen = len(seq_ep)
    start_pos = count*BATCH_SIZE%plen
    end_pos = min((count+1)*BATCH_SIZE%plen, plen)
    
    if end_pos <= start_pos:
        ebatch_p = np.concatenate([seq_ep[start_pos:plen], seq_ep[0:end_pos]])
        pbatch_p = np.concatenate([seq_pp[start_pos:plen], seq_pp[0:end_pos]])
    else:
        ebatch_p = seq_ep[start_pos: end_pos]
        pbatch_p = seq_pp[start_pos: end_pos]
    
    
    #negative set
    nlen = len(seq_en)
    start_pos = count*BATCH_SIZE%nlen
    end_pos = min((count+1)*BATCH_SIZE%nlen, nlen)
    
    if end_pos <= start_pos:
        ebatch_n = np.concatenate([seq_en[start_pos:nlen], seq_en[0:end_pos]])
        pbatch_n = np.concatenate([seq_pn[start_pos:nlen], seq_pn[0:end_pos]])
    else:
        ebatch_n = seq_en[start_pos: end_pos]
        pbatch_n = seq_pn[start_pos: end_pos]
    
    seq_e = np.concatenate([ebatch_p, ebatch_n])
    seq_p = np.concatenate([pbatch_p, pbatch_n])
    y = np.concatenate([np.ones([len(ebatch_p),1]), np.zeros([len(ebatch_n),1])])
    
    return seq_e, seq_p, y
    
 
def num_iteration(num):
    if num%100 == 0:
        return num//100
    else:
        return num//100+1

def cal_overlap(a,b):
    res = []
    for i in range(b.shape[0]):
        res.append(max(0, min(a[i,1], b[i,1]) - max(a[i,0], b[i,0]))/(b[i,1]-b[i,0]))
    return res 


def get_overlap(e_coor, p_coor, e_, p_, es_mean, es_sd, ee_mean, ee_sd, ps_mean, ps_sd, pe_mean, pe_sd):
    res1 = cal_overlap(np.column_stack((e_coor[:,0]*es_sd+es_mean, e_coor[:,1]*ee_sd+ee_mean)),e_)
    res2 = cal_overlap(np.column_stack((p_coor[:,0]*ps_sd+ps_mean, p_coor[:,1]*pe_sd+pe_mean)),p_)
    return res1, res2
    

def weight_variable(shape, name):
	initial = tf.truncated_normal(shape, stddev = 0.1)
	return tf.Variable(initial, name = name)


def bias_variable(shape, name):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name = name)



############################################################################################################
count = -1

xe_ = tf.placeholder(tf.float32, shape = (None, 4, enhancer_length,1))
xp_ = tf.placeholder(tf.float32, shape = (None, 4, promoter_length,1))
xe_coor_ = tf.placeholder(tf.float32, shape = (None, 2))
xp_coor_ = tf.placeholder(tf.float32, shape = (None, 2))
y_ =tf.placeholder(tf.float32, shape = (None,1))
bn_phase = tf.placeholder(tf.bool)
d_phase = tf.placeholder(tf.bool)


#enhancer: first convolution layer
e_W_conv1 = weight_variable([4, e_conv_width[0], 1, num_filters[0]], 'e_W_conv1')
	
e_layer0 = tf.nn.conv2d(xe_, e_W_conv1, strides = [1,1,1,1], padding='VALID')#+e_b_conv1
e_layer0_bn = e_layer0
e_layer1_pre = tf.nn.relu(e_layer0_bn)
e_layer1_pool = tf.nn.max_pool(e_layer1_pre, ksize = (1,1,pool_width,1), strides = (1,1, pool_width,1), padding='SAME')
e_layer1 = tf.layers.dropout(e_layer1_pool, rate = dropout_rate_cnn, training = d_phase)


#promoter: first convolution layer
p_layer0 = tf.nn.conv2d(xp_, e_W_conv1, strides = [1,1,1,1], padding='VALID')#+e_b_conv1
p_layer0_bn = p_layer0
p_layer1_pre = tf.nn.relu(p_layer0_bn)
p_layer1_pool = tf.nn.max_pool(p_layer1_pre, ksize = (1,1,pool_width,1) , strides = (1,1,pool_width,1), padding='SAME')
p_layer1 = tf.layers.dropout(p_layer1_pool, rate = dropout_rate_cnn, training = d_phase)



e_layer2 = tf.squeeze(e_layer1, 1)
p_layer2 = tf.squeeze(p_layer1, 1)
e_length = (enhancer_length-e_conv_width[0]+1)//pool_width+1
p_length = (promoter_length-e_conv_width[0]+1)//pool_width+1

v_matrix_2 = weight_variable([atten_hyper,1],'v_matrix_2') 
e_matrix_2 = weight_variable([num_filters[0], atten_hyper], 'e_matrix_2')
p_matrix_2 = weight_variable([num_filters[0], atten_hyper], 'p_matrix_2')
v_matrix_4 = weight_variable([atten_hyper,1],'v_matrix_4') 
e_matrix_4 = weight_variable([num_filters[0], atten_hyper], 'e_matrix_4')
p_matrix_4 = weight_variable([num_filters[0], atten_hyper], 'p_matrix_4')


p_layer4 = tf.transpose(p_layer2, perm = [0,2,1])
e_layer4 = tf.transpose(e_layer2, perm = [0,2,1])


e_layer3_pre_2 = tf.matmul(tf.reshape(e_layer2, shape = [-1, num_filters[0]]), e_matrix_2)
p_layer3_pre_2 = tf.matmul(tf.reshape(p_layer2, shape = [-1, num_filters[0]]), p_matrix_2)
e_layer3_2 = tf.reshape(e_layer3_pre_2, shape = [-1, e_length, atten_hyper])
p_layer3_2 = tf.reshape(p_layer3_pre_2, shape = [-1, p_length, atten_hyper])

e_layer3_pre_4 = tf.matmul(tf.reshape(e_layer2, shape = [-1, num_filters[0]]), e_matrix_4)
p_layer3_pre_4 = tf.matmul(tf.reshape(p_layer2, shape = [-1, num_filters[0]]), p_matrix_4)
e_layer3_4 = tf.reshape(e_layer3_pre_4, shape = [-1, e_length, atten_hyper])
p_layer3_4 = tf.reshape(p_layer3_pre_4, shape = [-1, p_length, atten_hyper])




#generate score matrix and softmax
###################################################################################
a_outer_layer_2 = tensor_array_ops.TensorArray(dtype = tf.float32, size = e_length, dynamic_size = False, infer_shape = True, tensor_array_name = 'a_outer_layer_2')

def a_inner_body_2(i,j, _layer):
    _layer = _layer.write(j, tf.matmul(tf.tanh(e_layer3_2[:,i,:]+p_layer3_2[:,j,:]), v_matrix_2) )
    return i, j+1, _layer

a_inner_cond_2 = lambda _1,j,_2: tf.less(j, p_length)

def a_one_enhancer(i):
    a_inner_layer_2 = tensor_array_ops.TensorArray(dtype = tf.float32, size = p_length, dynamic_size = False, infer_shape = True)
    _1, _2, r = tf.while_loop(cond = a_inner_cond_2, body = a_inner_body_2, loop_vars = (i,tf.constant(0,dtype=tf.int32) ,a_inner_layer_2))
    one_enhancer = r.gather(range(p_length))
    one_enhancer_score = tf.nn.softmax(one_enhancer, dim = 0)
    return one_enhancer_score


def a_outer_body_2(i,_layer):
    _layer = _layer.write(i, a_one_enhancer(i))
    return i+1, _layer
    
a_outer_cond_2 = lambda i,_1: tf.less(i, e_length)

_, A1_pre_2 = tf.while_loop(cond = a_outer_cond_2, body = a_outer_body_2, loop_vars = (tf.constant(0, dtype=tf.int32), a_outer_layer_2))


###################################################################################
a_outer_layer_4 = tensor_array_ops.TensorArray(dtype = tf.float32, size = p_length, dynamic_size = False, infer_shape = True, tensor_array_name = 'a_outer_layer_4')

def a_inner_body_4(i,j, _layer):
    _layer = _layer.write(j, tf.matmul(tf.tanh(e_layer3_4[:,j,:]+p_layer3_4[:,i,:]), v_matrix_4) )
    return i, j+1, _layer

a_inner_cond_4 = lambda _1,j,_2: tf.less(j, e_length)

def a_one_promoter(i):
    a_inner_layer_4 = tensor_array_ops.TensorArray(dtype = tf.float32, size = e_length, dynamic_size = False, infer_shape = True)
    _1, _2, r = tf.while_loop(cond = a_inner_cond_4, body = a_inner_body_4, loop_vars = (i, tf.constant(0, dtype = tf.int32), a_inner_layer_4))
    one_promoter = r.gather(range(e_length))
    one_promoter_score = tf.nn.softmax(one_promoter, dim = 0)
    return one_promoter_score

def a_outer_body_4(i, _layer):
    _layer = _layer.write(i, a_one_promoter(i))
    return i+1, _layer

a_outer_cond_4 = lambda i, _1: tf.less(i, p_length)

_, A1_pre_4 =tf.while_loop(cond = a_outer_cond_4, body = a_outer_body_4, loop_vars = (tf.constant(0, dtype = tf.int32), a_outer_layer_4))



#e_length, p_length, batch_size, 1
#batch_size, e_length, p_length, 1
A1_2 = tf.transpose(A1_pre_2.gather(range(e_length)), perm = [2,0,1,3])
A1_4 = tf.transpose(A1_pre_4.gather(range(p_length)), perm = [2,1,0,3])


#artificial convolution on promoter
#A1, p_layer4
###################################################################################
p_outer_layer = tensor_array_ops.TensorArray(dtype = tf.float32, size = e_length, dynamic_size = False, infer_shape = True, tensor_array_name = 'p_outer_layer')
   
def p_outer_body(i,_layer):
    _layer = _layer.write(i, tf.matmul(p_layer4,A1_2[:,i,:,:]))
    return i+1, _layer

p_outer_cond = lambda i, _1: tf.less(i, e_length)

_, p_layer5_pre = tf.while_loop(cond = p_outer_cond, body = p_outer_body, loop_vars = (tf.constant(0,dtype=tf.int32), p_outer_layer))

p_layer5 = p_layer5_pre.gather(range(e_length))
p_layer6_shrink = tf.squeeze(p_layer5, axis=3)
p_layer6 = tf.transpose(p_layer6_shrink, perm = [1,2,0])


#A1, e_layer4
###################################################################################
e_outer_layer = tensor_array_ops.TensorArray(dtype = tf.float32, size = p_length, dynamic_size = False, infer_shape = True, tensor_array_name = 'e_outer_layer')

def e_outer_body(i, _layer):
    _layer = _layer.write(i, tf.matmul(e_layer4, A1_4[:,:,i,:]))
    return i+1, _layer

e_outer_cond = lambda i, _1: tf.less(i, p_length)
_, e_layer5_pre = tf.while_loop(cond = e_outer_cond, body = e_outer_body, loop_vars = (tf.constant(0, dtype = tf.int32), e_outer_layer))

e_layer5 = e_layer5_pre.gather(range(p_length))
e_layer6_shrink = tf.squeeze(e_layer5, axis = 3)
e_layer6 = tf.transpose(e_layer6_shrink, perm = [1,0,2])


#align with e_layer4 and p_layer6
batch_num = tf.shape(p_layer6)[0]
e_layer7 = tf.reshape(e_layer6, shape = [-1, num_filters[0]])


#coordinate prediction
###################################################################################
e_coor1 = tf.reshape(e_layer6, shape = (-1, p_length*num_filters[0]))
p_coor1 = tf.reshape(p_layer6, shape = (-1, e_length*num_filters[0]))

e_W_coor1 = weight_variable([p_length*num_filters[0],dense_neuron_coor[0]], 'e_W_coor1')
e_b_coor1 = bias_variable([dense_neuron_coor[0]], 'e_b_coor1')
p_W_coor1 = weight_variable([e_length*num_filters[0],dense_neuron_coor[0]], 'p_W_coor1')
p_b_coor1 = bias_variable([dense_neuron_coor[0]], 'p_b_coor1')

e_coor2 = tf.matmul(e_coor1, e_W_coor1)+e_b_coor1
p_coor2 = tf.matmul(p_coor1, p_W_coor1)+p_b_coor1
e_coor2_bn = e_coor2
p_coor2_bn = p_coor2
e_coor3_pre = tf.nn.relu(e_coor2_bn)
p_coor3_pre = tf.nn.relu(p_coor2_bn)

e_coor3 = e_coor3_pre
p_coor3 = p_coor3_pre


e_W_coor2 = weight_variable([dense_neuron_coor[0],dense_neuron_coor[1]], 'e_W_coor2')
e_b_coor2 = bias_variable([dense_neuron_coor[1]], 'e_b_coor2')
p_W_coor2 = weight_variable([dense_neuron_coor[0],dense_neuron_coor[1]], 'p_W_coor2')
p_b_coor2 = bias_variable([dense_neuron_coor[1]], 'p_b_coor2')

e_coor4_pre = tf.nn.relu(tf.matmul(e_coor3, e_W_coor2)+e_b_coor2)
p_coor4_pre = tf.nn.relu(tf.matmul(p_coor3, p_W_coor2)+p_b_coor2)
e_coor4_bn = e_coor4_pre
p_coor4_bn = p_coor4_pre
e_coor4 = e_coor4_bn
p_coor4 = p_coor4_bn


e_W_coor3 = weight_variable([dense_neuron_coor[1],2], 'e_W_coor3')
e_b_coor3 = bias_variable([2], 'e_b_coor3')
p_W_coor3 = weight_variable([dense_neuron_coor[1],2], 'p_W_coor3')
p_b_coor3 = bias_variable([2], 'p_b_coor3')

e_coor5 = tf.matmul(e_coor4, e_W_coor3)+e_b_coor3
p_coor5 = tf.matmul(p_coor4, p_W_coor3)+p_b_coor3


dis_y_coor = tf.reduce_mean(tf.square(e_coor5-xe_coor_))+tf.reduce_mean(tf.square(p_coor5-xp_coor_))



#interaction matrix
#######################################################################################################
W1 = weight_variable([num_filters[0], num_filters[0], inter_dim[0]], 'W1')
b1 = bias_variable([inter_dim[0]], 'b1')

inter_layer = tensor_array_ops.TensorArray(dtype = tf.float32, size = inter_dim[0], dynamic_size = False, infer_shape = True, tensor_array_name = 'interaction_layer')

def inter_body(i, _layer):
    left_inter = tf.matmul(e_layer7, W1[:,:,i])
    left_inter = tf.reshape(left_inter, shape = [batch_num, p_length, num_filters[0]])
    right_inter = tf.matmul(left_inter, p_layer6)
    right_inter = tf.reshape(right_inter, shape = [batch_num,p_length*e_length])
    right_inter = right_inter+b1[i]    
    _layer = _layer.write(i, right_inter)
    return i+1, _layer

inter_cond = lambda i, _1: tf.less(i, inter_dim[0])

_, inter_layer0 = tf.while_loop(cond = inter_cond, body = inter_body, loop_vars = (tf.constant(0, dtype = tf.int32), inter_layer))
    

layer0 = tf.transpose(inter_layer0.gather(range(inter_dim[0])),perm = [1,0,2])
layer0_norm = layer0

#top-k max pooling
layer1 = tf.nn.top_k(layer0_norm, k = topk).values
layer2 = tf.reshape(layer1, shape = (batch_num, inter_dim[0]*topk))


W_dense1 = weight_variable([inter_dim[0]*topk,dense_neuron], 'W_dense1')
b_dense1 = bias_variable([dense_neuron], 'b_dense1')
layer3 = tf.nn.relu(tf.matmul(layer2, W_dense1)+b_dense1)

layer3_bn = layer3
layer4 = tf.layers.dropout(layer3_bn, rate = dropout_rate, training = d_phase)


W_dense2 = weight_variable([dense_neuron,1], 'W_dense2')
b_dense2 = bias_variable([1], 'b_dense2')
dis_y = tf.matmul(layer4, W_dense2)+b_dense2


dis_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dis_y, labels=y_))
dis_loss = lamb*dis_cross_entropy+dis_y_coor

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
if update_ops:
    with tf.control_dependencies(update_ops):
        dis_train_step = tf.train.AdamOptimizer(1e-3).minimize(dis_loss)
else:
    dis_train_step = tf.train.AdamOptimizer(1e-3).minimize(dis_loss)



#laod the data
############################################################################################################
seq_ep, seq_en, seq_pp, seq_pn = input_sequences(efile = file_pre+'_enhancer.fasta', pfile = file_pre+'_promoter.fasta', labelfile = file_pre+'_label.txt')


seq_ep_t, seq_en_t, seq_pp_t, seq_pn_t = input_sequences(efile =file_pre+'_enhancer_test.fasta', pfile = file_pre+'_promoter_test.fasta', labelfile = file_pre+'_label_test.txt')


#load coordinates
es_mean, es_sd, ee_mean, ee_sd, ps_mean, ps_sd, pe_mean, pe_sd = input_stat(efile = file_pre+'_enhancer_coor_stat.txt', pfile = file_pre+'_promoter_coor_stat.txt')
    
seq_ep_coor, seq_en_coor, seq_pp_coor, seq_pn_coor = input_coordinates(efile = file_pre+'_enhancer_coor.txt', pfile = file_pre+'_promoter_coor.txt', labelfile = file_pre+'_label.txt')

seq_ep_t_coor, seq_en_t_coor, seq_pp_t_coor, seq_pn_t_coor = input_coordinates(efile = file_pre+'_enhancer_coor_test.txt', pfile = file_pre+'_promoter_coor_test.txt', labelfile = file_pre+'_label_test.txt')



############################################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()


entropy_loss = []
reg_loss = []
F1_accu = []
AUC_accu = []
AUPR_accu = []
count_epoch = 0


batch_per_epoch = (seq_ep.shape[0]+seq_en.shape[0])/(BATCH_SIZE*2)
num_of_batches = batch_per_epoch*num_of_epoch
print 'num_of_batches:', num_of_batches


ptm = time.time()

for i in range(num_of_batches):
    if (i %batch_per_epoch==0) and (i>0):
        perm = np.random.permutation(seq_ep.shape[0])
        seq_ep = seq_ep[perm]
        perm = np.random.permutation(seq_en.shape[0])
        seq_en = seq_en[perm]
        perm = np.random.permutation(seq_pp.shape[0])
        seq_pp = seq_pp[perm]
        perm = np.random.permutation(seq_pn.shape[0])
        seq_pn = seq_pn[perm]
        
        if (i % (10*batch_per_epoch) == 0):
            count_epoch = count_epoch+10
            saver.save(sess, 'output/'+script_id+'/'+str(count_epoch))
        
    count = count+1
    ebatch, pbatch, ybatch = get_batches(seq_ep, seq_en, seq_pp, seq_pn, count, BATCH_SIZE)
    ebatch_coor, pbatch_coor = get_coordinates_batches(seq_ep_coor, seq_en_coor, seq_pp_coor, seq_pn_coor, count, BATCH_SIZE)
    if (i%100==0) or (i == num_of_batches-1):
        entropy_loss.append(dis_cross_entropy.eval(feed_dict = {xe_: ebatch, xp_:pbatch, y_:ybatch, d_phase:0, bn_phase: 0}, session = sess))
        reg_loss.append(dis_y_coor.eval(feed_dict = {xe_: ebatch, xp_:pbatch, xe_coor_:ebatch_coor, xp_coor_:pbatch_coor, d_phase:0, bn_phase: 0}, session = sess))
        
    dis_train_step.run(feed_dict = {xe_: ebatch, xp_:pbatch, xe_coor_:ebatch_coor, xp_coor_:pbatch_coor,y_:ybatch, d_phase:1, bn_phase: 1}, session = sess)
	
    if (i%output_step==0) or (i==num_of_batches-1):
        print(i)
        
        positive_set = []
        for j in range(num_iteration(seq_ep_t.shape[0])):
            start_pos = j*100
            end_pos = min((j+1)*100, seq_ep_t.shape[0])
            temp_pos = tf.nn.sigmoid(dis_y.eval(feed_dict = {xe_: seq_ep_t[start_pos:end_pos], xp_: seq_pp_t[start_pos:end_pos], d_phase:0, bn_phase: 0}, session = sess))
            positive_set.extend(temp_pos.eval(session = sess).flatten().tolist())
	
	
        negative_set = []
        for j in range(num_iteration(seq_en_t.shape[0])):
            start_pos = j*100
            end_pos = min((j+1)*100,seq_en_t.shape[0])
            temp_neg = tf.nn.sigmoid(dis_y.eval(feed_dict = {xe_: seq_en_t[start_pos:end_pos], xp_: seq_pn_t[start_pos:end_pos], d_phase:0, bn_phase: 0}, session = sess))
            negative_set.extend(temp_neg.eval(session = sess).flatten().tolist())
    
        #F1 score calculation
        positive_set = np.array(positive_set)
        negative_set = np.array(negative_set)
        F1_thre = 0.5
        TP = sum(positive_set > F1_thre)
        FN = sum(positive_set < F1_thre)
        FP = sum(negative_set > F1_thre)
        F1 = (2.0*TP)/(2*TP+FP+FN)
        F1_accu.append(F1)


        #AUROC, AUPR calculation
        y = np.concatenate([np.ones([len(positive_set),1]), np.zeros([len(negative_set),1])]).flatten()
        x = np.concatenate([positive_set,negative_set])
        AUC = metrics.roc_auc_score(y,x)
        AUPR = metrics.average_precision_score(y,x)
        AUC_accu.append(AUC)
        AUPR_accu.append(AUPR)
        print F1
        print AUC
        print AUPR
        
	


############################################################################################################
seconds = time.time()-ptm
fen, miao = divmod(seconds, 60)
xiao, fen = divmod(fen, 60)
print "%d:%02d:%02d" % (xiao, fen, miao)


saver.save(sess, 'output/'+script_id+'/model')

variable_filename = 'output/'+script_id+'/variables.out'


#save certain variables
with open(variable_filename, 'w') as fout:
    pickle.dump([entropy_loss, reg_loss, F1_accu,AUC_accu, AUPR_accu], fout)




