#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-12-11 下午7:09
# @Author  : xujun
# @Email   : xujunrt@163.com
# @Version : 1.0

from utility import save_embedding

class TrainHelper():
    @staticmethod
    def helper(num_epoch, dataset, bpr_optimizer,
               # pp_sampler,
               # pd_sampler,
               dd_sampler,
               dt_sampler,
               # djconf_sampler,
               # dorg_sampler,
               # dyear_sampler,
               dabstract_sampler,
               eval_f1, sampler_method,filename):

        bpr_optimizer.init_model(dataset)
        if sampler_method == "uniform":
            for _ in xrange(0, num_epoch):
                bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    # for i, j, t in pp_sampler.generate_triplet_uniform(dataset):
                    #     bpr_optimizer.update_pp_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                    # for i, j, t in pd_sampler.generate_triplet_uniform(dataset):
                    #     bpr_optimizer.update_pd_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    for i, j, t in dd_sampler.generate_triplet_uniform(dataset):
                        bpr_optimizer.update_dd_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

                    for i, j, t in dt_sampler.generate_triplet_uniform(dataset):
                        bpr_optimizer.update_dt_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dt_loss(i, j, t)

                    # for i, j, t in djconf_sampler.generate_triplet_uniform(dataset):
                    #     bpr_optimizer.update_djconf_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_djconf_loss(i, j, t)
                    #
                    # for i, j, t in dorg_sampler.generate_triplet_uniform(dataset):
                    #     bpr_optimizer.update_dorg_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dorg_loss(i, j, t)
                    #
                    # for i, j, t in dyear_sampler.generate_triplet_uniform(dataset):
                    #     bpr_optimizer.update_dyear_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dyear_loss(i, j, t)

                    for i, j, t in dabstract_sampler.generate_triplet_uniform(dataset):
                        bpr_optimizer.update_dabstract_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dabstract_loss(i, j, t)

                average_loss = float(bpr_loss) / dataset.num_nnz
                # print "average bpr loss is " + str(average_loss)

                #average_loss = float(bpr_loss) / dataset.num_nnz
                #print "average bpr loss is " + str(average_loss)
            average_f1,average_pre,average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
                #print 'f1 is ' + str(average_f1)
            save_embedding(bpr_optimizer.paper_latent_matrix,
                           dataset.paper_list, bpr_optimizer.latent_dimen, filename)
            return average_f1,average_pre,average_rec
        elif sampler_method == "reject":
            for _ in xrange(0, num_epoch):
                bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    # for i, j, t in pp_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                    #     bpr_optimizer.update_pp_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)
                    #
                    # for i, j, t in pd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                    #     bpr_optimizer.update_pd_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    for i, j, t in dd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                        bpr_optimizer.update_dd_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

                    for i, j, t in dt_sampler.generate_triplet_reject(dataset,bpr_optimizer):
                        bpr_optimizer.update_dt_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dt_loss(i, j, t)

                    # for i, j, t in djconf_sampler.generate_triplet_reject(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_djconf_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_djconf_loss(i, j, t)
                    #
                    # for i, j, t in dorg_sampler.generate_triplet_reject(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_dorg_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dorg_loss(i, j, t)

                    # for i, j, t in dyear_sampler.generate_triplet_reject(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_dyear_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dyear_loss(i, j, t)

                    for i, j, t in dabstract_sampler.generate_triplet_reject(dataset,bpr_optimizer):
                        bpr_optimizer.update_dabstract_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dabstract_loss(i, j, t)

            average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
            # print 'f1 is ' + str(average_f1)
            return average_f1,average_pre,average_rec
        elif sampler_method == "adaptive":
            for _ in xrange(0, num_epoch):
                bpr_loss = 0.0
                for _ in xrange(0, dataset.num_nnz):
                    """
                    update embedding in person-person network
                    update embedding in person-document network
                    update embedding in doc-doc network
                    """
                    # for i, j, t in pp_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                    #     bpr_optimizer.update_pp_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)
                    #
                    # for i, j, t in pd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                    #     bpr_optimizer.update_pd_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                    for i, j, t in dd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                        bpr_optimizer.update_dd_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)
                    for i, j, t in dt_sampler.generate_triplet_adaptive(dataset,bpr_optimizer):
                        bpr_optimizer.update_dt_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dt_loss(i, j, t)

                    # for i, j, t in djconf_sampler.generate_triplet_adaptive(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_djconf_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_djconf_loss(i, j, t)
                    #
                    # for i, j, t in dorg_sampler.generate_triplet_adaptive(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_dorg_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dorg_loss(i, j, t)

                    # for i, j, t in dyear_sampler.generate_triplet_adaptive(dataset,bpr_optimizer):
                    #     bpr_optimizer.update_dyear_gradient(i, j, t)
                    #     bpr_loss += bpr_optimizer.compute_dyear_loss(i, j, t)

                    for i, j, t in dabstract_sampler.generate_triplet_adaptive(dataset,bpr_optimizer):
                        bpr_optimizer.update_dabstract_gradient(i, j, t)
                        bpr_loss += bpr_optimizer.compute_dabstract_loss(i, j, t)

            average_f1, average_pre, average_rec = eval_f1.compute_f1(dataset, bpr_optimizer)
            # print 'f1 is ' + str(average_f1)
            return average_f1,average_pre,average_rec
        save_embedding(bpr_optimizer.paper_latent_matrix,
                       dataset.paper_list, bpr_optimizer.latent_dimen,filename)
