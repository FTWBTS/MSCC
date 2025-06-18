Optimal_Multi_algo_HP_dict = {
    'IForest': {'n_estimators': 25, 'max_features': 0.8},
    'LOF': {'n_neighbors': 50, 'metric': 'euclidean'},    
    'PCA': {'n_components': 0.25},        
    'HBOS': {'n_bins': 30, 'tol': 0.5},
    'COPOD': {'n_jobs':1},    
    'CBLOF': {'n_clusters': 4, 'alpha': 0.6},
    'EIF': {'n_trees': 50},   
    'CNN': {'window_size': 512, 'num_channel': [32, 32, 32]},
    'MSCC':{'window_size': 512, 'num_channel': [32, 32, 32],'kernel_size':4, 'stride' : 1, 'dropout_rate':0.1,'scale1':20,'scale2':100,'scale3':200,'init_alpha':1.0,'lamda':1e-13},#PSM
    # 'MSCC':{'window_size': 512, 'num_channel': [32, 32, 32],'kernel_size':4, 'stride' : 1, 'dropout_rate':0.1,'scale1':40,'scale2':100,'scale3':200,'init_alpha':1.0,'lamda':1e-13},#PSM_dialet 
    # 'MSCC':{'window_size': 512, 'num_channel': [32, 32, 40],'kernel_size':4, 'stride' : 1, 'dropout_rate':0.2,'scale1':185,'scale2':278,'scale3':357,'init_alpha':3.0,'lamda':1e-14},#MSL
    # 'MSCC':{'window_size': 720, 'num_channel': [32, 32, 40],'kernel_size':3, 'stride' : 1, 'dropout_rate':0.25,'scale1':200,'scale2':357,'scale3':434,'init_alpha':1.0,'lamda':1e-14},#SWaT
    # 'MSCC':{'window_size': 512, 'num_channel': [32, 32, 40],'kernel_size':8, 'stride' : 1, 'dropout_rate':0.35,'scale1':285,'scale2':313,'scale3':322,'init_alpha':2.0,'lamda':-0.01},#SMD
    'TFMAE': {'window_size': 500}, 
    'LSTMAD': {'window_size': 150, 'lr': 0.0008},  
    'TranAD': {'win_size': 10, 'lr': 0.001}, 
    'AnomalyTransformer': {'win_size': 50, 'lr': 0.001},  
    'OmniAnomaly': {'win_size': 100, 'lr': 0.002},
    'USAD': {'win_size':100, 'lr': 0.001},  #原来是100，0.001
    'TimesNet': {'win_size': 96, 'lr': 0.0001},
    'FITS': {'win_size': 100, 'lr': 0.001},
    'MatrixProfile': {'periodicity': 1},
}
