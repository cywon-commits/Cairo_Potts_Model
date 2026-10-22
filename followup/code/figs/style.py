import os as _os; ROOT=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
C=['#2a78d6','#eb6834','#1baf7a','#eda100','#e87ba4','#008300']
INK='#222222'; MUTED='#777777'; GRID='#dddddd'
plt.rcParams.update({'font.size':9.5,'axes.edgecolor':MUTED,'axes.labelcolor':INK,'xtick.color':MUTED,'ytick.color':MUTED,
 'axes.spines.top':False,'axes.spines.right':False,'lines.linewidth':2,'legend.frameon':False,'axes.titlesize':10,'figure.dpi':170})
HEX=[(1,0),(1,1),(0,1),(-1,0),(-1,-1),(0,-1)]
SC=ROOT
