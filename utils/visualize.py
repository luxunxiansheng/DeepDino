import pandas as pd

class Visual(object):
       
    
    @staticmethod
    def visual_mean_scores(score_file_path):
        start = 0
        interval = 10        
        scores_df = pd.read_csv(score_file_path)
        mean_scores = pd.DataFrame(columns=['score'])
        while interval <= len(scores_df):
            mean_scores.loc[len(mean_scores)] = scores_df.loc[start:interval].mean()['scores']
            start = interval
            interval = interval + 10
        
        mean_scores.plot()     

    @staticmethod
    def visual_max_scores(score_file_path):
        start = 0
        interval = 10        
        scores_df = pd.read_csv(score_file_path)
        max_scores = pd.DataFrame(columns=['max_score'])
        while interval <= len(scores_df):
            max_scores.loc[len(max_scores)] = scores_df.loc[start:interval].max()['scores']
            start = interval
            interval = interval + 10
           
        max_scores.plot()   

    @staticmethod
    def visual_q_max_scores(q_value_file_path):
        start = 0
        interval = 10        
        q_max_df = pd.read_csv(q_value_file_path)
        q_max_scores = pd.DataFrame(columns=['q_max'])
        while interval <= len(q_max_df):
            q_max_scores.loc[len(q_max_scores)] = q_max_df.loc[start:interval].max()['actions']
            start = interval
            interval = interval + 1000

        q_max_scores.plot()   


