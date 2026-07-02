
import os
from pathlib import Path
import numpy as np
from astropy.stats import RipleysKEstimator
import time as t
from upmask_core import outer
from upmask_core.dataIO import readFiles, readINI, dread, dmask, dxynorm, dwrite
import multiprocessing as mp


def main(
    parallel_flag, parallel_procs, rnd_seed, verbose, ID_c, x_c, y_c,
    data_cols, data_errs, oultr_method, stdRegion_nstd, OL_runs, resampleFlag,
    PCAflag, PCAdims, GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs,
        N_cl_max, clust_method, clRjctMethod, C_thresh, cl_method_pars):
    """
    """
    out_folder = "output"
    # Create 'output' folder if it does not exist
    Path('./{}'.format(out_folder)).mkdir(parents=True, exist_ok=True)

    # Process all files inside the '/input' folder
    inputfiles = readFiles()

    for file_path in inputfiles:

        print("\n")
        print("===========================================================")
        print("Processing         : {}".format(file_path.name))
        # Set a random seed for reproducibility
        if rnd_seed == 'None':
            seed = np.random.randint(100000)
        else:
            seed = int(rnd_seed)
        print("Random seed        : {}".format(seed))
        np.random.seed(seed)

        # Original data
        full_data, cl_ID, cl_xy, cl_data, cl_errs, data_rjct = dread(
            file_path, ID_c, x_c, y_c, data_cols, data_errs)

        # Remove outliers
        msk_data, ID, xy, data, data_err = dmask(
            cl_ID, cl_xy, cl_data, cl_errs, oultr_method, stdRegion_nstd)

        # Normalize (x, y) data to [0, 1]
        xy01 = dxynorm(xy)

        probs_all = dataProcess(
            ID, xy01, data, data_err, verbose, OL_runs,
            parallel_flag, parallel_procs, resampleFlag, PCAflag, PCAdims,
            GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max,
            clust_method, clRjctMethod, C_thresh, cl_method_pars)

        if OL_runs > 1:
            # Obtain the mean of all runs. This are the final probabilities
            # assigned to each star in the frame
            probs_mean = np.mean(probs_all, 0)
        else:
            probs_mean = probs_all[0]

        # Write final data to file
        dwrite(
            out_folder, file_path, full_data, msk_data, data_rjct, probs_mean)


def dataProcess(
    ID, xy, data, data_err, verbose, OL_runs, parallel_flag,
    parallel_procs, resampleFlag, PCAflag, PCAdims, GUMM_flag, GUMM_perc,
    KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method, clRjctMethod,
        C_thresh, cl_method_pars):
    """
    """
    start_t = t.time()

    # TODO this should be handled by the logging() module
    # Set print() according to the 'verbose' parameter
    if verbose == 0:
        prfl = open(os.devnull, 'w')
    else:
        prfl = None

    # Print input parameters to screen
    if parallel_flag:
        print("Parallel runs      : {}".format(parallel_flag))
        print("Processes          : {}".format(parallel_procs))
    print("Outer loop runs    : {}".format(OL_runs))
    if PCAflag:
        print("Apply PCA          : {}".format(PCAflag))
        print(" PCA N_dims        : {}".format(PCAdims))
    if GUMM_flag:
        print("Apply GUMM         : {}".format(GUMM_flag))
        print(" GUMM percentile   : {}".format(GUMM_perc))
    if KDEP_flag:
        print("Obtain KDE probs   : {}".format(KDEP_flag))

    print("Inner loop runs    : {}".format(IL_runs))
    print("Stars per cluster  : {}".format(N_membs))
    print("Maximum clusters   : {}".format(N_cl_max))
    print("Clustering method  : {}".format(clust_method))
    if cl_method_pars:
        for key, val in cl_method_pars.items():
            print(" {:<17} : {}".format(key, val))
    print("")
    # print("Rejection method   : {}".format(clRjctMethod))
    # if clRjctMethod != 'rkfunc':
    #     print("Threshold          : {:.2f}".format(C_thresh))

    # Define RK test with an area of 1.
    # Kest = None
    # if clRjctMethod == 'rkfunc':
    Kest = RipleysKEstimator(area=1, x_max=1, y_max=1, x_min=0, y_min=0)
    # if clRjctMethod == 'kdetest' or clust_method == 'rkmeans':
    #     from rpy2.robjects import r
    #     from rpy2.robjects import numpy2ri
    #     from rpy2.robjects.packages import importr
    #     # cat(paste("R version: ",R.version.string,"\n"))
    #     importr('MASS')
    #     r("""
    #     set.seed(12345)
    #     """)
    #     numpy2ri.activate()
    #     r.assign('nruns', 2000)
    #     r.assign('nKde', 50)

    # Arguments for the Outer Loop
    OLargs = (
        ID, xy, data, data_err, resampleFlag, PCAflag, PCAdims, GUMM_flag,
        GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max, clust_method,
        clRjctMethod, Kest, C_thresh, cl_method_pars, prfl)

    # TODO: Breaks if verbose=0
    if parallel_flag is True:
        if parallel_procs == 'None':
            # Use *almost* all the cores
            N_cpu = mp.cpu_count() - 1
        else:
            N_cpu = int(parallel_procs)
        with mp.Pool(processes=N_cpu) as p:
            manager = mp.Manager()
            KDE_vals = manager.dict({})
            probs_all = p.starmap(
                OLfunc, [(OLargs, KDE_vals) for _ in range(OL_runs)])

    else:
        KDE_vals = {}
        probs_all = []
        for _ in range(OL_runs):
            print("\n--------------------------------------------------------")
            print("OL run {}".format(_ + 1))
            # The KDE_vals dictionary is updated after each OL run
            probs, KDE_vals = outer.loop(*OLargs, KDE_vals)
            probs_all.append(probs)

            p_dist = [
                (np.mean(probs_all, 0) > _).sum() for _ in
                (.5, .75, .9, .95, .99)]
            print("\nP>(.5, .75, .9, .95, .99): {}, {}, {}, {}, {}".format(
                *p_dist), file=prfl)

    elapsed = t.time() - start_t
    if elapsed > 60.:
        elapsed, ms_id = elapsed / 60., "minutes"
    else:
        ms_id = "seconds"
    print("\nTime consumed: {:.1f} {}".format(elapsed, ms_id))

    return probs_all


def OLfunc(args, KDE_vals):
    """
    Here to handle the parallel runs.
    """
    probs, _ = outer.loop(*args, KDE_vals)
    return probs


# =====================================================================
# 🛸 针对 hunt24-audit 工厂管线定制的现代化内存解耦 API 入口
# =====================================================================
def upmask_api(input_df, custom_config, rnd_seed=None):
    """
    在内存中直接运行 UPMASK 的闭环函数。
    
    :param input_df: pandas.DataFrame, 必须包含恒星ID、位置(x, y)及用于聚类的特征列和误差列
    :param custom_config: dict, 替换 params.ini 的参数字典
    :param rnd_seed: int/str, 随机种子。如果为 None, 则使用全局默认设置
    :return: numpy.ndarray, 返回与 input_df 行数完全对应的最终成员概率数组
    """
    import numpy as np
    
    # 1. 提取字典参数并设置默认值（防止字典里漏写某些不常用的参数）
    parallel_flag   = custom_config.get('parallel_flag', False)
    parallel_procs  = custom_config.get('parallel_procs', 'None')
    verbose         = custom_config.get('verbose', 1)
    ID_c            = custom_config.get('ID_c')            # ID 列名
    x_c             = custom_config.get('x_c')             # x 坐标列名
    y_c             = custom_config.get('y_c')             # y 坐标列名
    data_cols       = custom_config.get('data_cols')       # 聚类特征列名列表 (如 pmra, pmdec)
    data_errs       = custom_config.get('data_errs')       # 聚类误差列名列表
    oultr_method    = custom_config.get('oultr_method', 'stdregion')
    stdRegion_nstd  = custom_config.get('stdRegion_nstd', 3.0)
    OL_runs         = custom_config.get('OL_runs', 5)
    resampleFlag    = custom_config.get('resampleFlag', True)
    PCAflag         = custom_config.get('PCAflag', False)
    PCAdims         = custom_config.get('PCAdims', 'None')
    GUMM_flag       = custom_config.get('GUMM_flag', False)
    GUMM_perc       = custom_config.get('GUMM_perc', 10)
    KDEP_flag       = custom_config.get('KDEP_flag', False)
    IL_runs         = custom_config.get('IL_runs', 100)
    N_membs         = custom_config.get('N_membs', 10)
    N_cl_max        = custom_config.get('N_cl_max', 20)
    clust_method    = custom_config.get('clust_method', 'kmeans')
    clRjctMethod    = custom_config.get('clRjctMethod', 'uniform')
    C_thresh        = custom_config.get('C_thresh', 0.5)
    cl_method_pars  = custom_config.get('cl_method_pars', {})

    # 2. 随机种子初始化
    if rnd_seed is None or rnd_seed == 'None':
        seed = np.random.randint(100000)
    else:
        seed = int(rnd_seed)
    np.random.seed(seed)

    # 3. 仿照 dread() 逻辑，直接从内存中的 DataFrame 提取 numpy 数组
    # 提取完整 ID, xy坐标, 数据和误差
    cl_ID = input_df[ID_c].to_numpy()
    cl_xy = input_df[[x_c, y_c]].to_numpy()
    cl_data = input_df[data_cols].to_numpy()
    cl_errs = input_df[data_errs].to_numpy()

    # 4. 绕过原文件 I/O，复用原作者的核心清洗、转换与计算流程
    from upmask_core.dataIO import dmask, dxynorm
    
    # 剔除离群值 (注意: 原作者 dmask 会返回被过滤掉的 mask 数据，我们这里主要拿有效数组)
    msk_data, ID, xy, data, data_err = dmask(
        cl_ID, cl_xy, cl_data, cl_errs, oultr_method, stdRegion_nstd)

    # 将坐标 (x, y) 归一化到 [0, 1] 空间（Ripley's K 估算器需要）
    xy01 = dxynorm(xy)

    # 5. 调用原作者的进程处理核心
    probs_all = dataProcess(
        ID, xy01, data, data_err, verbose, OL_runs,
        parallel_flag, parallel_procs, resampleFlag, PCAflag, PCAdims,
        GUMM_flag, GUMM_perc, KDEP_flag, IL_runs, N_membs, N_cl_max,
        clust_method, clRjctMethod, C_thresh, cl_method_pars)

    # 6. 计算最终概率均值
    if OL_runs > 1:
        probs_mean = np.mean(probs_all, 0)
    else:
        probs_mean = probs_all[0]

    # 7. 将计算出来的有效恒星概率，精准映射回原始 input_df 的物理长度中
    # 因为原作者的 dmask 会过滤掉一部分离群星，我们需要让返回数组的长度和原本传入的行数完全一致
    final_probabilities = np.zeros(len(input_df))
    
    # 利用 ID 进行高效索引匹配映射
    id_to_prob = dict(zip(ID, probs_mean))
    for idx, row_id in enumerate(cl_ID):
        final_probabilities[idx] = id_to_prob.get(row_id, 0.0)  # 被丢弃的离群星概率默认为 0.0

    return final_probabilities



if __name__ == '__main__':

    # # Limit numpy's cores used to 1
    # # Source: https://stackoverflow.com/a/58195413/1391441, also
    # # https://stackoverflow.com/q/17053671/1391441

    # parallel_flag, parallel_procs = params[:2]
    # if parallel_flag:
    #     if parallel_procs == 'None':
    #         # Use *almost* all the cores
    #         parallel_procs = mp.cpu_count() - 1
    #     else:
    #         # Never use more than these cores
    #         parallel_procs = min(int(parallel_procs), mp.cpu_count() - 1)
    # else:
    #     parallel_procs = 1

    # Read input parameters.
    params = readINI()

    if params[0] is False:
        # Disable numpy's multithreading
        parallel_procs = str(1)
        os.environ["OMP_NUM_THREADS"] = parallel_procs
        os.environ["MKL_NUM_THREADS"] = parallel_procs
        os.environ["OPENBLAS_NUM_THREADS"] = parallel_procs
        os.environ["VECLIB_MAXIMUM_THREADS"] = parallel_procs
        os.environ["NUMEXPR_NUM_THREADS"] = parallel_procs
    else:
        # If numpy is allowed to multithread, disable the parallel run
        params[1] = False

    main(*params[1:])
