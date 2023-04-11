import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

# 创建地图对象
map = Basemap()

# 绘制海岸线
map.drawcoastlines()

# 显示地图
plt.show()