#--------------------------
# Modified from Song et al.
#--------------------------

import numpy as np
import math

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, 
                 target_train_performance, target_test_performance, num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        '''
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        
        self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_loss = self._loss_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_loss = self._loss_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_loss = self._loss_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_loss = self._loss_comp(self.t_te_outputs, self.t_te_labels)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, int(self.s_tr_labels[i])] for i in range(len(self.s_tr_labels))])
        self.s_te_conf = np.array([self.s_te_outputs[i, int(self.s_te_labels[i])] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, int(self.t_tr_labels[i])] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, int(self.t_te_labels[i])] for i in range(len(self.t_te_labels))])
        
        self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        self.s_te_entr = self._entr_comp(self.s_te_outputs)
        self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    def _loss_comp(self, probs, y):
        return -np.sum(np.multiply(np.array([y, 1-y]).T, self._log_value(probs)), axis=1)
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(len(tr_values)+0.0)
            te_ratio = np.sum(te_values<value)/(len(te_values)+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
        return thre
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, s_tr_attr, s_te_attr, t_tr_values, t_te_values, t_tr_attr, t_te_attr):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        s_tr_mem, s_te_non_mem = 0, 0
        num_attributes = max(s_tr_attr) + 1
        for a in range(num_attributes):
            t_tr_mem_a, t_te_non_mem_a = 0, 0
            
            s_tr_values_w_attr = s_tr_values[s_tr_attr==a]
            s_tr_labels_w_attr = self.s_tr_labels[s_tr_attr==a]
            
            s_te_values_w_attr = s_te_values[s_te_attr==a]
            s_te_labels_w_attr = self.s_te_labels[s_te_attr==a]
            
            t_tr_values_w_attr = t_tr_values[t_tr_attr==a]
            t_tr_labels_w_attr = self.t_tr_labels[t_tr_attr==a]
            
            t_te_values_w_attr = t_te_values[t_te_attr==a]
            t_te_labels_w_attr = self.t_te_labels[t_te_attr==a]
            total_a_tr = len(t_tr_labels_w_attr)
            total_a_te = len(t_te_labels_w_attr)
            for num in range(self.num_classes):
                thre = self._thre_setting(s_tr_values_w_attr[s_tr_labels_w_attr==num], s_te_values_w_attr[s_te_labels_w_attr==num])
                t_tr_mem += np.sum(t_tr_values_w_attr[t_tr_labels_w_attr==num]>=thre)
                t_te_non_mem += np.sum(t_te_values_w_attr[t_te_labels_w_attr==num]<thre)
                
                s_tr_mem += np.sum(s_tr_values_w_attr[s_tr_labels_w_attr==num]>=thre)
                s_te_non_mem += np.sum(s_te_values_w_attr[s_te_labels_w_attr==num]<thre)
                
                t_tr_mem_a += np.sum(t_tr_values_w_attr[t_tr_labels_w_attr==num]>=thre)
                t_te_non_mem_a += np.sum(t_te_values_w_attr[t_te_labels_w_attr==num]<thre)
            mem_inf_acc_a = 0.5*(t_tr_mem_a/(total_a_tr+0.0) + t_te_non_mem_a/(total_a_te+0.0))
            print('For membership inference attack via {n}, group attr {a}, the attack acc is {acc:.3f}'.format(n=v_name,a=a, acc=mem_inf_acc_a))
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return
    
    def _mem_inf_thre_no_class(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        t_tr_mem, t_te_non_mem = 0, 0
        for num in range(self.num_classes):
            thre = self._thre_setting(s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num])
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return
    
    def _mem_inf_benchmarks(self, s_tr_attr, s_te_attr, t_tr_attr, t_te_attr, all_methods=True, benchmark_methods=[]):
        if (all_methods) or ('correctness' in benchmark_methods):
            self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, s_tr_attr, s_te_attr, self.t_tr_conf, self.t_te_conf, t_tr_attr, t_te_attr)
        if (all_methods) or ('entropy' in benchmark_methods):
            self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, s_tr_attr, s_te_attr, -self.t_tr_entr, -self.t_te_entr, t_tr_attr, t_te_attr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, s_tr_attr, s_te_attr, -self.t_tr_m_entr, -self.t_te_m_entr, t_tr_attr, t_te_attr)
        if (all_methods) or ('loss' in benchmark_methods):
            self._mem_inf_thre('loss', self.s_tr_loss, self.s_te_loss, s_tr_attr, s_te_attr, self.t_tr_loss, self.t_te_loss, t_tr_attr, t_te_attr)
        return
