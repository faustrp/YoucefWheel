
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:10:52 2022
@author: apple
"""
# from fibermagic.IO.NeurophotometricsIO import read_project_rawdata, read_project_logs
import plotly.graph_objects as go
from pathlib import Path
import plotly.express as px
from fibermagictmaze.core.demodulate import zdFF_airPLS, add_zdFF
from fibermagictmaze.core.perievents import perievents
import os
import pandas as pd
from fibermagictmaze.IO.NeurophotometricsIO import extract_leds, read_project_logs
import numpy as np

pd.options.plotting.backend = "plotly"
from plotly.offline import iplot

import plotly
from plotly.offline import plot
from plotly.subplots import make_subplots

import plotly.io as pio
pio.kaleido.scope.default_format = "svg"

from plotly.graph_objs.layout import YAxis,XAxis

import numpy

from scipy import stats
from scipy import signal


#NPM_RED = 560
NPM_GREEN = 470
NPM_ISO = 410


def read_project_rawdata(project_path, subdirs, data_file, ignore_dirs=['meta']):
    """
    Runs standard zdFF processing pipeline on every trial, mouse and sensor and puts data together in a single df
    :param ignore_dirs: array of directories to exclude from reading
    :param subdirs: array with description of subdirectories to be a column
    :param project_path: root path of the project
    :param data_file: name of each file containing raw data
    :return: dataframe with the following multicolumn/index structure
                            Signal      Reference
    Trial 1 mouse A 560     0.1         0.5
                    560     0.25        0.56
                    470     0.13        0.57
            mouse B 560    ...         ...
    Trial 2 mouse A 470    ...
            mouse B 560
    ...     ...
    """
    # walk through all specified subdirectories
    dfs = list()

    def recursive_listdir(path, levels):
        if levels:
            for dir in os.listdir(path):
                if dir in ignore_dirs:
                    continue
                recursive_listdir(path / dir, levels - 1)
        else:
            print(path / data_file)
            df = pd.read_csv(path / data_file)
            region_to_mouse = pd.read_csv(path / 'region_to_mouse.csv')
            if 'Flags' in df.columns:  # legacy fix: Flags were renamed to LedState
                df = df.rename(columns={'Flags': 'LedState'})

            df = extract_leds(df).dropna()
            # dirty hack to come around dropped frames until we find better solution -
            # it makes about 0.16 s difference
            df = df.iloc[1:]
            df.FrameCounter = np.arange(0, len(df)) // len(df.wave_len.unique())
            df = df.set_index('FrameCounter')
            #df = df[~df.index.duplicated()]
            regions = [column for column in df.columns if 'Region' in column]
            for region in regions:
                channel = NPM_GREEN #if 'G' in region else NPM_RED
                sdf = pd.DataFrame(data={
                    **{subdirs[i]: path.parts[- (len(subdirs) - i)] for i in range(len(subdirs))},
                    'Mouse': region_to_mouse[region_to_mouse.region == region].mouse.values[0],
                    'Wheel': region_to_mouse[region_to_mouse.region == region].wheel.values[0],
                    'Channel': channel,
                    'Signal': df[region][df.wave_len == channel],
                    'Reference': df[region][df.wave_len == NPM_ISO]
                }
                )
                dfs.append(sdf)
    recursive_listdir(Path(project_path), len(subdirs))
    df = pd.concat(dfs)
    df = df.reset_index().set_index([*subdirs, 'Mouse','Wheel', 'Channel', 'FrameCounter'])
    print(df)
    return df




PROJECT_PATH = Path(r'/Users/You.B/Desktop/Operant YB')



df_idx = read_project_rawdata(PROJECT_PATH,
                              [], 'FP.csv', ignore_dirs=['meta', 'processed', 'processed.zip'])
print ("DF_IDX")
print (df_idx)

'''
df_idx = df_idx.reset_index()

fig = px.line(df_idx[df_idx.Channel==470], x='FrameCounter', y=['Signal','Reference'], template="simple_white")


#fig.update_yaxes(range=[-1.5, 6])


fig.update_layout(
           yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze Raw data Mouse#2981',
    hovermode="x"
    )

plot(fig)
'''

df_idx = add_zdFF(df_idx, smooth_win=25, remove=200, lambd=5e5).set_index('FrameCounter', append=True)
'''

fig = px.line(df_idx[df_idx.Channel==560], x='FrameCounter', y='zdFF (airPLS)', template="simple_white")

#fig = px.line(peri[peri.Channel==560], x='Timestamp', y='zdFF (airPLS)', color='Trial',color_discrete_sequence=px.colors.sequential.Blues, template="simple_white")

#fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', color='Trial', facet_col='Mouse', facet_col_wrap='Trial')

#fig.update_yaxes(range=[-1.5, 6])

fig.update_layout(
 #   yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze FedArmIN Mouse#2980',
    hovermode="x"
)

plot(fig)

'''











def sync_from_pc_time(logs, path):
    """
    attatches Bonsai frame numbers to the the logs
    :param logs: log df with column SI@0.0
    :param sync_signals: df with timestamps of recorded sync signals from FP, columns Item1 and Item2
    :param timestamps: df with timestamp for each FP frame, columns Item1 and Item2
    :return: log file with new column Frame_Bonsai with FP frame number for each event
    """
    timestamps = pd.read_csv(path / 'time.csv')
#    logs['Event'] = logs[logs['WheelStart']==1]['WheelStop'] & logs[logs['WheelStop']==1]['WheelStop']
    # join FP SI with logs
    # convert Bonsai Timestamps to Frame number
    logs['FrameCounter'] = timestamps.Item2.searchsorted(logs.Timestamp)
#    logs = logs[['FrameCounter', 'Event', 'Timestamp']]
    logs = logs[['FrameCounter', 'Timestamp', 'WheelStart', 'WheelStop','WheelAngleIncr']]

    logs = logs.dropna()
    logs['FrameCounter'] //= 2
    return logs.set_index('FrameCounter')

logs = read_project_logs(PROJECT_PATH, [], sync_fun=sync_from_pc_time,
                         ignore_dirs=['meta', 'processed', 'images'])



'''
peri = perievents(df_idx, logs, 5, 60)

peri = peri.reset_index()


print (df_idx)

print (peri.columns)

print (peri)












if not os.path.exists("images"):
    os.mkdir("images")

colorscales = px.colors.named_colorscales()

#fig = px.line(df_idx[df_idx.Channel==560], x='Timestamp', y='zdFF (airPLS)', template="simple_white")

fig = px.line(peri[peri.Channel==560], x='Timestamp', y='zdFF (airPLS)', color='Trial',color_discrete_sequence=px.colors.sequential.Inferno, template="simple_white")

fig.update_xaxes(range=[-5, 5])
fig.update_yaxes(range=[-2, 4])

fig.update_layout(
    yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze FedArmIN Mouse#2984',
    hovermode="x"
)


fig.write_image("images/fig1.png",width=1200, height=600, scale=4)


fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', color='Trial',color_discrete_sequence=px.colors.sequential.Inferno, template="simple_white")

fig.update_xaxes(range=[-5, 5])
fig.update_yaxes(range=[-3, 4])

fig.update_layout(
    yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze FedArmIN Mouse#2984',
    hovermode="x"
)
fig.write_image("images/fig2.png",width=1200, height=600, scale=4)









#fig = px.line(peri[peri.Channel==560], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap=6)
fig = px.line(peri[peri.Channel==560], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap=6, facet_row_spacing = 0.02)

#fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap='Trial')

fig.update_xaxes(range=[-5, 5])
fig.update_yaxes(range=[-2, 3])

fig.update_layout(
    yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze FedArmIN Mouse#2984',
    hovermode="x"
)

fig.write_image("images/fig3.png",width=1000, height=600, scale=4)]



fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap=6, facet_row_spacing = 0.02)
#fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap=6 )

#fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', facet_col='Trial', facet_col_wrap='Trial')

fig.update_xaxes(range=[-5, 5])
fig.update_yaxes(range=[-2, 2])

fig.update_layout(
    yaxis_title='zdFF',
    title='Fiberphotometry Dopamine/ACh T-Maze FedArmIN Mouse#2984',
    hovermode="x"
)

fig.write_image("images/fig4.png",width=1000, height=600, scale=4)




peri_stat = peri.groupby(['Channel', 'Timestamp'])['zdFF (airPLS)'].describe().reset_index()
peri_sem = peri.groupby(['Channel', 'Timestamp'])['zdFF (airPLS)'].sem().rename('SEM')
peri_sem = peri_sem.to_frame()
'''

'''
#peri_RefStat = peri.groupby(['Channel','Timestamp'])['Reference'].describe().reset_index()
#peri_RefSem = peri.groupby(['Channel','Timestamp'])['Reference'].sem().rename('REFSEM')
#peri_RefSem = peri_RefSem.to_frame()


#print (peri_stat)
#print(peri_sem)

extract = peri_sem['SEM']
extract = extract.to_frame()
extract.astype(float)
#extract.reset_index()
#extract2 = extract.loc[extract['Timestamp'] == -4]
#print (extract)


peri_stat.reset_index(drop=True)
peri_stat = peri_stat.merge(extract, on=('Channel', 'Timestamp'))

#print (peri_stat.columns.tolist())
#fig = px.line(peri_stat, x='Timestamp', y='mean', error_y = 'SEM')



#periRW0= peri_stat([peri_stat.Channel==560] & [peri_stat.Wheel=='W0'])
#periGW0= peri_stat([peri_stat.Channel==470] & [peri_stat.Wheel=='W0'])
#periRW1= peri_stat([peri_stat.Channel==560] & [peri_stat.Wheel=='W1'])
#periGW1= peri_stat([peri_stat.Channel==470] & [peri_stat.Wheel=='W1'])

#periR = peri_stat[peri_stat.Channel==560]
periG = peri_stat[peri_stat.Channel==470]

fig = go.Figure([

    go.Scatter(
        name='5-HT3.0',
        x=periG['Timestamp'],
        y=periG['mean'],
        mode='lines',
        line=dict(color='green', width=4),
#        facet_col='Mouse',
    ),
    go.Scatter(
        name='Upper Bound',
        x=periG['Timestamp'],
        y=periG['mean']+periG['SEM'],
        mode='lines',
        marker=dict(color="green"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=periG['Timestamp'],
        y=periG['mean']-periG['SEM'],
        marker=dict(color="green"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(0,100,80,0.2)',
        fill='tonexty',
        showlegend=False
    ),
    
])
fig.update_layout(
    xaxis_title='time(sec)',
    yaxis_title='z-scores',
    title='5-HT3.0 Wheel/FED3',
    hovermode="x",
    template="simple_white"
)
#fig.show()
fig.update_xaxes(range=[-5, 5])
#fig.update_yaxes(range=[-1.5, 7])
fig.add_vline(x=0, line_width=5, line_dash="dash", line_color='rgb(37,37,37,1)')
fig.update_layout(font_size=38)
fig.write_image("images/fig5.png",width=2000, height=1600, scale=8)







df_idx = df_idx.reset_index()


fig = px.line(df_idx[df_idx.Channel==560], x='FrameCounter', y='zdFF (airPLS)', template="simple_white")

#fig = px.line(peri[peri.Channel==560], x='Timestamp', y='zdFF (airPLS)', color='Trial',color_discrete_sequence=px.colors.sequential.Blues, template="simple_white")

#fig = px.line(peri[peri.Channel==470], x='Timestamp', y='zdFF (airPLS)', color='Trial', facet_col='Mouse', facet_col_wrap='Trial')


fig.update_xaxes(range=[0, 35000])
#fig.update_yaxes(range=[-1.5, 7])
fig.update_layout(
 #   yaxis_title='zdFF',
    title='Fiberphotometry Dopamine T-Maze Z-Score',
    hovermode="x"
)

fig.write_image("images/fig6.png",width=1000, height=600, scale=4)
'''















WheelRunning = pd.DataFrame()
WheelRunning = pd.read_csv(PROJECT_PATH / 'B2537WheelDay1.log')






print (WheelRunning)
df_idx = df_idx.reset_index()
Plot1X = df_idx[df_idx.Channel==470]


'''
fig = make_subplots(rows=1, cols=2, specs = [[{"secondary_y": True}, {"secondary_y": True}]],
                    horizontal_spacing=0.2,
                    shared_xaxes = False)

# Subplot 1
## Line


fig.update_yaxes(range = [0, 6], dtick = 0.1, secondary_y=False)
fig.add_trace(
    go.Scatter(x= Plot1X['FrameCounter'], y=Plot1X['zdFF (airPLS)'], mode='lines'),
    row = 1, col = 1
)



# Subplot 2
## Line
fig.update_yaxes(range = [-1000, 800], dtick = 0.1, secondary_y=True)
fig.add_trace(
    go.Scatter(x=WheelRunning['Timestamp'], mode='lines'),secondary_y=True,
    row = 1, col = 1
)

fig.add_trace(
    go.Scatter(y = WheelRunning['WheelAngle'], mode='lines'),
    row = 1, col = 1
)


fig.update_layout(xaxis2= {'anchor': 'y', 'overlaying': 'x', 'side': 'top'})
#fig.WheelRunning.update(xaxis='x2')
fig.update_layout(xaxis2_range=[0,35000])

plot(fig)
'''

'''
WheelRunning["START"] = (WheelRunning["OnWheel"] > 0).map({True: "Positive", False: "Negative"})
WheelRunning["STARTDet"] = WheelRunning["START"].shift() != WheelRunning["START"]
'''

'''
## add a secondary xaxis and yaxis in the layout
layout = go.Layout(
    xaxis2 = XAxis( 
        overlaying='x',
        side='top',
    ),
    yaxis2=YAxis(
        overlaying='y',
        side='right',
    ),

    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.83,
        font=dict(
            family="Trebuchet",
            size=20,
            color="black"
         ),
            bgcolor="LightGray",
            bordercolor="Black",
            borderwidth=0
    )
)

fig = px.line(df_idx[df_idx.Channel==560], x='FrameCounter', y='zdFF (airPLS)')

#fig.add_trace(go.Scatter(x=WheelRunning['Timestamp'], y=WheelRunning['OnWheel'], xaxis='x2', yaxis='y2'))

#fig.add_trace(go.Scatter(x=WheelRunning['Timestamp'], y=WheelRunning['OnWheel'], xaxis='x2', yaxis='y2'))
#fig.update_traces(name='Points', showlegend = True,mode='lines')


## add the final trace specifying the secondary x-axis and secondary y-axis
#fig.add_trace(go.Scatter(x=df3['C'], y=df3['D'], xaxis='x2', yaxis='y2'))

fig.layout = layout
'''

logs = logs.reset_index()

fig = px.line(df_idx[df_idx.Channel==470], x='FrameCounter', y='zdFF (airPLS)')
fig.add_trace(
    go.Scatter(x= logs['FrameCounter'], y=logs['WheelAngleIncr']/2000, mode='lines',line=dict(color="black")),
)
TimeStart = logs[(logs.WheelStart == 1)]
TimeStop = logs[(logs.WheelStop == 1)]
TimeStart = TimeStart.reset_index()
TimeStop = TimeStop.reset_index()
Start = []
Stop = []
Start.append(TimeStart.FrameCounter)
Stop.append(TimeStop.FrameCounter)
df_Start=pd.DataFrame(Start)
df_Stop=pd.DataFrame(Stop)
WSta = df_Start.iloc[0]
WSto = df_Stop.iloc[0]
count = 0

for x in df_Stop:
    WSta = df_Start.iloc[0,count]
    WSto = df_Stop.iloc[0,count]
    fig.add_vrect(x0=WSta, x1=WSto , y0 = 0.45, y1 = 0.55, fillcolor='red',  opacity=0.35,  line_width=0.2)#layer="below",

    count+=1


'''
count2 = 0    
TimePellet = logs[(logs.Reward==1)]
TimePellet = TimePellet.reset_index()
StartPellet = []
StartPellet.append(TimePellet.FrameCounter)
df_StartPellet=pd.DataFrame(StartPellet)
print (df_StartPellet)
PSta = df_StartPellet.iloc[0]
if len(StartPellet):
    for x in df_StartPellet:
        PSta = df_StartPellet.iloc[0,count2]
        fig.add_vline(x= PSta, line_width=1.5, line_dash="dash", line_color='rgb(37,37,37,1)')
        #fig.add_vline(x= PSta,y0 = 0.45, y1 = 0.55, line_width=1.5, line_dash="dash", line_color='rgb(37,37,37,1)')
    
        count2+=1
'''
'''
count3 = 0    
TimeFedArm = logs[(logs.FedArmIN==1)]
TimeFedArm = TimeFedArm.reset_index()
StartFedArm = []
StartFedArm.append(TimeFedArm.FrameCounter)
df_StartFedArm=pd.DataFrame(StartFedArm)
#print (df_StartFedArm)
FSta = df_StartFedArm.iloc[0]
if len(StartFedArm):
    for x in df_StartFedArm:
        FSta = df_StartFedArm.iloc[0,count3]
        fig.add_vline(x= FSta,y0 = 0.45, y1 = 0.55, line_width=3, line_dash="dash", line_color='orange')
        #fig.add_vline(x= PSta,y0 = 0.45, y1 = 0.55, line_width=1.5, line_dash="dash", line_color='rgb(37,37,37,1)')
    
        count3+=1

    

TimeStart2 = logs[(logs.MiddleArm == 1)]
TimeStop2 = logs[(logs.MiddleArm == 1)]
TimeStart2 = TimeStart2.reset_index()
TimeStop2 = TimeStop2.reset_index()
Start2 = []
Stop2 = []
Start2.append(TimeStart2.FrameCounter)
Stop2.append(TimeStop2.FrameCounter)
df_Start2=pd.DataFrame(Start2)
df_Stop2=pd.DataFrame(Stop2)
MSta = df_Start2.iloc[0]
MSto = df_Stop2.iloc[0]
count2 = 0

for y in df_Start2:
    MSta = df_Start2.iloc[0,count2]
    MSto = df_Stop2.iloc[0,count2]
    fig.add_vrect(x0=MSta, x1=MSto , y0 = 0.25, y1 = 0.35, fillcolor='orange',  opacity=0.35,  line_width=0.2)#layer="below",

    count2+=1

'''
'''

TimeStart = WheelRunning[(WheelRunning.WheelStart == 1)]
TimeStop = WheelRunning[(WheelRunning.WheelStop == 1)]
Start = []
Stop = []
Start.append(TimeStart.Timestamp)
Stop.append(TimeStop.Timestamp)
df_Start=pd.DataFrame(Start)
df_Stop=pd.DataFrame(Stop)
count = 0
fig.layout = layout
for x in WheelRunning:
    
    WSta = df_Start.iloc[0,count]
    WSto = df_Stop.iloc[0,count]
    
    fig.add_vrect(x0=WSta, x1=WSto, 
                  fillcolor='red', opacity=0.3, layer="below", line_width=0)
    count+=1

'''


plot(fig)



    






'''
subfig = make_subplots(specs=[[{"secondary_x": True}]])

# create two independent figures with px.line each containing data from multiple columns
fig = px.line(df_idx[df_idx.Channel==560], x=df_idx[df_idx.Channel==560]['FrameCounter'], render_mode="webgl",)
fig2 = px.line(WheelRunning, x=WheelRunning['Timestamp'], render_mode="webgl",)

fig2.update_traces(xaxis="x2")



subfig.add_traces(fig.data + fig2.data)
subfig.layout.xaxis.title="Timestamp"
subfig.layout.xaxis2.title="Wheel"
#subfig.layout.yaxis2.type="log"
#subfig.layout.yaxis2.title="Log Y"
# recoloring is necessary otherwise lines from fig und fig2 would share each color
# e.g. Linear-, Log- = blue; Linear+, Log+ = red... we don't want this
subfig.for_each_trace(lambda t: t.update(line=dict(color=t.marker.color)))
plot(subfig)

'''

'''

fig = px.line(df_idx[df_idx.Channel==470], x='FrameCounter', y='Signal')
fig.add_trace(
    go.Scatter(x= logs['FrameCounter'], y=logs['WheelAngle']/2000, mode='lines',line=dict(color="black")),
)
TimeStart = logs[(logs.WheelStart == 1)]
TimeStop = logs[(logs.WheelStop == 1)]
TimeStart = TimeStart.reset_index()
TimeStop = TimeStop.reset_index()
Start = []
Stop = []
Start.append(TimeStart.FrameCounter)
Stop.append(TimeStop.FrameCounter)
df_Start=pd.DataFrame(Start)
df_Stop=pd.DataFrame(Stop)
WSta = df_Start.iloc[0]
WSto = df_Stop.iloc[0]
count = 0

for x in df_Stop:
    WSta = df_Start.iloc[0,count]
    WSto = df_Stop.iloc[0,count]
    fig.add_vrect(x0=WSta, x1=WSto , y0 = 0.45, y1 = 0.55, fillcolor='red',  opacity=0.35,  line_width=0.2)#layer="below",

    count+=1

TimeStart2 = logs[(logs.MiddleArm == 1)]
TimeStop2 = logs[(logs.MiddleArm == 1)]
TimeStart2 = TimeStart2.reset_index()
TimeStop2 = TimeStop2.reset_index()
Start2 = []
Stop2 = []
Start2.append(TimeStart2.FrameCounter)
Stop2.append(TimeStop2.FrameCounter)
df_Start2 = pd.DataFrame(Start2)
df_Stop2 = pd.DataFrame(Stop2)
MSta = df_Start2.iloc[0]
MSto = df_Stop2.iloc[0]
count2 = 0

for y in df_Start2:
    MSta = df_Start2.iloc[0, count2]
    MSto = df_Stop2.iloc[0, count2]
    fig.add_vrect(x0=MSta, x1=MSto, y0=0.25, y1=0.35, fillcolor='orange', opacity=0.35,
                  line_width=0.2)  # layer="below",

    count2 += 1



TimeStart = WheelRunning[(WheelRunning.WheelStart == 1)]
TimeStop = WheelRunning[(WheelRunning.WheelStop == 1)]
Start = []
Stop = []
Start.append(TimeStart.Timestamp)
Stop.append(TimeStop.Timestamp)
df_Start = pd.DataFrame(Start)
df_Stop = pd.DataFrame(Stop)
count = 0
fig.layout = layout
for x in WheelRunning:
    WSta = df_Start.iloc[0, count]
    WSto = df_Stop.iloc[0, count]

    fig.add_vrect(x0=WSta, x1=WSto,
                  fillcolor='red', opacity=0.3, layer="below", line_width=0)
    count += 1




count3 = 0
TimeFedArm = logs[(logs.FedArmIN == 1)]
TimeFedArm = TimeFedArm.reset_index()
StartFedArm = []
StartFedArm.append(TimeFedArm.FrameCounter)
df_StartFedArm = pd.DataFrame(StartFedArm)
# print (df_StartFedArm)
FSta = df_StartFedArm.iloc[0]
if len(StartFedArm):
    for x in df_StartFedArm:
        FSta = df_StartFedArm.iloc[0, count3]
        fig.add_vline(x=FSta, y0=0.45, y1=0.55, line_width=3, line_dash="dash", line_color='orange')
        # fig.add_vline(x= PSta,y0 = 0.45, y1 = 0.55, line_width=1.5, line_dash="dash", line_color='rgb(37,37,37,1)')

        count3 += 1
plot(fig)
'''