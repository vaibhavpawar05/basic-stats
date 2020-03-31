# roll up at custid, subcatid, periodid level
    
app_reg_txns_hist2_rl = app_reg_txns_hist2.groupby(['custid','subcatid','periodid'], as_index=False).\
                        agg({'transamount':{'atxn':np.sum},\
                        'subcategory':{'ntxn':np.size}})
                        
# Flatten index
col_names = [t[0] if t[1]=='' else t[1] for t in app_reg_txns_hist2_rl.columns]    
app_reg_txns_hist2_rl.columns = col_names
