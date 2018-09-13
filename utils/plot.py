# #### BEGIN LICENSE BLOCK #####
# Version: MPL 1.1/GPL 2.0/LGPL 2.1
#
# The contents of this file are subject to the Mozilla Public License Version
# 1.1 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
# http://www.mozilla.org/MPL/
#
# Software distributed under the License is distributed on an "AS IS" basis,
# WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
# for the specific language governing rights and limitations under the
# License.
#
#
# Contributor(s):
#
#    Bin.Li (ornot2008@yahoo.com)
#
#
# Alternatively, the contents of this file may be used under the terms of
# either the GNU General Public License Version 2 or later (the "GPL"), or
# the GNU Lesser General Public License Version 2.1 or later (the "LGPL"),
# in which case the provisions of the GPL or the LGPL are applicable instead
# of those above. If you wish to allow use of your version of this file only
# under the terms of either the GPL or the LGPL, and not to allow others to
# use your version of this file under the terms of the MPL, indicate your
# decision by deleting the provisions above and replace them with the notice
# and other provisions required by the GPL or the LGPL. If you do not delete
# the provisions above, a recipient may use your version of this file under
# the terms of any one of the MPL, the GPL or the LGPL.
#
# #### END LICENSE BLOCK #####
#
# /


# Adapted from  https://github.com/openai/baselines/blob/master/baselines/results_plotter.py

import os
import re

import numpy as np
import pandas as pd



class LoadMonitorResultsError(Exception):
    pass

class Plotter:
    EXT="*score.csv"
    
    COLORS = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink',
              'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise',
              'darkgreen', 'tan', 'salmon', 'gold', 'lightpurple', 'darkred', 'darkblue']

    X_TIMESTEPS = 'timesteps'
    X_EPISODES = 'episodes'
    X_WALLTIME = 'walltime_hrs'

    def __init__(self):
        pass

    def _rolling_window(self, a, window):
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

    def _window_func(self, x, y, window, func):
        yw = self._rolling_window(y, window)
        yw_func = func(yw, axis=-1)
        return x[window - 1:], yw_func

    def _timestep2xy(self, ts, xaxis):
        if xaxis == Plotter.X_TIMESTEPS:
            x = np.cumsum(ts.l.values)
            y = ts.r.values
        elif xaxis == Plotter.X_EPISODES:
            x = np.arange(len(ts))
            y = ts.r.values
        elif xaxis == Plotter.X_WALLTIME:
            x = ts.t.values / 3600.
            y = ts.r.values
        else:
            raise NotImplementedError
        return x, y
    
    
    def load_monitor_log(self,monitor_files):
        dfs = []
        headers = []
        for fname in monitor_files:
            with open(fname, 'rt') as fh:
                    assert fname.endswith('csv')
                    firstline = fh.readline()
                    assert firstline[0] == 'Scores'
                    df = pd.read_csv(fh, index_col=None)
                    headers.append(header)
                                
                    df['t'] += header['t_start']
        
        dfs.append(df)
        df = pandas.concat(dfs)
        df.sort_values('t', inplace=True)
        df.reset_index(inplace=True)
        df['t'] -= min(header['t_start'] for header in headers)
        # df.headers = headers # HACK to preserve backwards compatibility
        return df
    
    
    def load_results(self, dirs, max_timesteps=1e8, x_axis=X_TIMESTEPS, episode_window=100):
        timesteplist = []
        for dir in dirs:
            ts = load_monitor_log(dir)
            ts = ts[ts.l.cumsum() <= max_timesteps]
            timesteplist.append(ts)
        xy_list = [self._timestep2xy(ts, x_axis) for ts in timesteplist]
        if episode_window:
            xy_list = [self._window_func(x, y, episode_window, np.mean) for x, y in xy_list]
        return xy_list

    def load_evaluation_episodes_results(self,dirs,evaluation_episodes_interval,evaluation_episodes,max_timesteps=1e8):
        raw_data = self.load_results(dirs, max_timesteps, episode_window=0)
        ys = []
        for x, y in raw_data:
            y = np.reshape(np.asarray(y), (-1, evaluation_episodes)).mean(-1)
            x = np.arange(y.shape[0]) * evaluation_episodes_interval
            ys.append(y)
        return x, np.stack(ys)

    def average(self, xy_list, bin, max_timesteps, top_k=0):
        if top_k:
            perf = [np.max(y) for _, y in xy_list]
            top_k_runs = np.argsort(perf)[-top_k:]
            new_xy_list = []
            for r, (x, y) in enumerate(xy_list):
                if r in top_k_runs:
                    new_xy_list.append((x, y))
            xy_list = new_xy_list
        new_x = np.arange(0, max_timesteps, bin)
        new_y = []
        for x, y in xy_list:
            new_y.append(np.interp(new_x, x, y))
        return new_x, np.asarray(new_y)

    def load_log_dirs(self, pattern, negative_pattern=' ', root='./log', **kwargs):
        dirs = [item[0] for item in os.walk(root)]
        leaf_dirs = []
        for i in range(len(dirs)):
            if i + 1 < len(dirs) and dirs[i + 1].startswith(dirs[i]):
                continue
            leaf_dirs.append(dirs[i])
        names = []
        p = re.compile(pattern)
        np = re.compile(negative_pattern)
        for dir in leaf_dirs:
            if p.match(dir) and not np.match(dir):
                names.append(dir)
                print(dir)

        return sorted(names)

    def plot_standard_error(self, data, x=None, **kwargs):
        import matplotlib.pyplot as plt
        if x is None:
            x = np.arange(data.shape[1])
        e_x = np.std(data, axis=0) / np.sqrt(data.shape[0])
        m_x = np.mean(data, axis=0)
        plt.plot(x, m_x, **kwargs)
        del kwargs['label']
        plt.fill_between(x, m_x + e_x, m_x - e_x, alpha=0.3, **kwargs)
