# 合肥地铁1号线合肥火车站——长淮——明光路——大东门——包公园——合工大南区——朱岗——秋浦河路——葛大店——望湖城——合肥南站——南站南广场——骆岗——高王——滨湖会展中心——紫庐——塘西河公园——金斗公园——云谷路——万达城——万年埠——丙子铺——九联圩
# 合肥地铁2号线南岗——桂庄——汽车西站——振兴路——蜀山西——大蜀山——天柱路——科学大道——十里庙——西七里塘——五里墩——三里庵——安农大——三孝口——四牌楼——大东门——三里街——东五里井——东七里——漕冲——东二十埠——龙岗——王岗——三十埠
# 合肥地铁3号线相城路——职教城东——职教城——幼儿师范——文浍苑——勤劳村——新海大道——窦桥湾——方庙——竹丝滩——合肥火车站——鸭林冲——淮南路——一里井——海棠——郑河——四泉桥——杏花村——合肥西站——南新庄——西七里塘——国防科技大学——洪岗——市政务中心——合肥大剧院——图书馆——省博物院——安医大二附院——繁华大道——大学城北——工大翡翠湖校区——安大磬苑校区——幸福坝
# 合肥地铁4号线青龙岗——合肥七中——量子科学中心——科大先研院——北雁湖——玉兰大道——金桂——柳树塘——图书馆——天鹅湖——天鹅湖东——姚公庙——南屏路——薛河——竹西——淝南——合肥南站——望湖城南——葛大店南——工经学院——尧渡河路——五里庙——唐桥——东七里——站塘——方庙——新海公园——十里村——陶冲湖东——安医大四附院——综保区
# 合肥地铁5号线望湖城西——合肥南站——盛大——包河苑——义兴——大连路——花园大道——黄河路——扬子江路——义城——省政务中心东——方兴湖——渡江纪念馆——沈湾——华山路——云谷路——清水冲——云川公园——滨湖竹园——贵阳路
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # 启用Tkinter作为图形界面的后端

plt.rcParams['font.sans-serif'] = ['KaiTi']     #设置字体为楷体
plt.rcParams['axes.unicode_minus']=False        #显示负号
# 创建一个空的无向图
G = nx.Graph()
# 添加1号线的边
G.add_edge('合肥火车站', '长淮', line='1号线')
G.add_edge('长淮', '明光路', line='1号线')
G.add_edge('明光路', '大东门', line='1号线')
G.add_edge('大东门', '包公园', line='1号线')
G.add_edge('包公园', '合工大南区', line='1号线')
G.add_edge('合工大南区', '朱岗', line='1号线')
G.add_edge('朱岗', '秋浦河路', line='1号线')
G.add_edge('秋浦河路', '葛大店', line='1号线')
G.add_edge('葛大店', '望湖城', line='1号线')
G.add_edge('望湖城', '合肥南站', line='1号线')
G.add_edge('合肥南站', '南站南广场', line='1号线')
G.add_edge('南站南广场', '骆岗', line='1号线')
G.add_edge('骆岗', '高王', line='1号线')
G.add_edge('高王', '滨湖会展中心', line='1号线')
G.add_edge('滨湖会展中心', '紫庐', line='1号线')
G.add_edge('紫庐', '塘西河公园', line='1号线')
G.add_edge('塘西河公园', '金斗公园', line='1号线')
G.add_edge('金斗公园', '云谷路', line='1号线')
G.add_edge('云谷路', '万达城', line='1号线')
G.add_edge('万达城', '万年埠', line='1号线')
G.add_edge('万年埠', '丙子铺', line='1号线')
G.add_edge('丙子铺', '九联圩', line='1号线')

# 添加2号线的边
G.add_edge('南岗', '桂庄', line='2号线')
G.add_edge('桂庄', '汽车西站', line='2号线')
G.add_edge('汽车西站', '振兴路', line='2号线')
G.add_edge('振兴路', '蜀山西', line='2号线')
G.add_edge('蜀山西', '大蜀山', line='2号线')
G.add_edge('大蜀山', '天柱路', line='2号线')
G.add_edge('天柱路', '科学大道', line='2号线')
G.add_edge('科学大道', '十里庙', line='2号线')
G.add_edge('十里庙', '西七里塘', line='2号线')
G.add_edge('西七里塘', '五里墩', line='2号线')
G.add_edge('五里墩', '三里庵', line='2号线')
G.add_edge('三里庵', '安农大', line='2号线')
G.add_edge('安农大', '三孝口', line='2号线')
G.add_edge('三孝口', '四牌楼', line='2号线')
G.add_edge('四牌楼', '大东门', line='2号线')
G.add_edge('大东门', '三里街', line='2号线')
G.add_edge('三里街', '东五里井', line='2号线')
G.add_edge('东五里井', '东七里', line='2号线')
G.add_edge('东七里', '漕冲', line='2号线')
G.add_edge('漕冲', '东二十埠', line='2号线')
G.add_edge('东二十埠', '龙岗', line='2号线')
G.add_edge('龙岗', '王岗', line='2号线')
G.add_edge('王岗', '三十埠', line='2号线')

# 添加3号线的边
G.add_edge('相城路', '职教城东', line='3号线')
G.add_edge('职教城东', '职教城', line='3号线')
G.add_edge('职教城', '幼儿师范', line='3号线')
G.add_edge('幼儿师范', '文浍苑', line='3号线')
G.add_edge('文浍苑', '勤劳村', line='3号线')
G.add_edge('勤劳村', '新海大道', line='3号线')
G.add_edge('新海大道', '窦桥湾', line='3号线')
G.add_edge('窦桥湾', '方庙', line='3号线')
G.add_edge('方庙', '竹丝滩', line='3号线')
G.add_edge('竹丝滩', '合肥火车站', line='3号线')
G.add_edge('合肥火车站', '鸭林冲', line='3号线')
G.add_edge('鸭林冲', '淮南路', line='3号线')
G.add_edge('淮南路', '一里井', line='3号线')
G.add_edge('一里井', '海棠', line='3号线')
G.add_edge('海棠', '郑河', line='3号线')
G.add_edge('郑河', '四泉桥', line='3号线')
G.add_edge('四泉桥', '杏花村', line='3号线')
G.add_edge('杏花村', '合肥西站', line='3号线')
G.add_edge('合肥西站', '南新庄', line='3号线')
G.add_edge('南新庄', '西七里塘', line='3号线')
G.add_edge('西七里塘', '国防科技大学', line='3号线')
G.add_edge('国防科技大学', '洪岗', line='3号线')
G.add_edge('洪岗', '市政务中心', line='3号线')
G.add_edge('市政务中心', '合肥大剧院', line='3号线')
G.add_edge('合肥大剧院', '图书馆', line='3号线')
G.add_edge('图书馆', '省博物院', line='3号线')
G.add_edge('省博物院', '安医大二附院', line='3号线')
G.add_edge('安医大二附院', '繁华大道', line='3号线')
G.add_edge('繁华大道', '大学城北', line='3号线')
G.add_edge('大学城北', '工大翡翠湖校区', line='3号线')
G.add_edge('工大翡翠湖校区', '安大磬苑校区', line='3号线')
G.add_edge('安大磬苑校区', '幸福坝', line='3号线')

# 添加4号线的边
G.add_edge('青龙岗', '合肥七中', line='4号线')
G.add_edge('合肥七中', '量子科学中心', line='4号线')
G.add_edge('量子科学中心', '科大先研院', line='4号线')
G.add_edge('科大先研院', '北雁湖', line='4号线')
G.add_edge('北雁湖', '玉兰大道', line='4号线')
G.add_edge('玉兰大道', '金桂', line='4号线')
G.add_edge('金桂', '柳树塘', line='4号线')
G.add_edge('柳树塘', '图书馆', line='4号线')
G.add_edge('图书馆', '天鹅湖', line='4号线')
G.add_edge('天鹅湖', '天鹅湖东', line='4号线')
G.add_edge('天鹅湖东', '姚公庙', line='4号线')
G.add_edge('姚公庙', '南屏路', line='4号线')
G.add_edge('南屏路', '薛河', line='4号线')
G.add_edge('薛河', '竹西', line='4号线')
G.add_edge('竹西', '淝南', line='4号线')
G.add_edge('淝南', '合肥南站', line='4号线')
G.add_edge('合肥南站', '望湖城南', line='4号线')
G.add_edge('望湖城南', '葛大店南', line='4号线')
G.add_edge('葛大店南', '工经学院', line='4号线')
G.add_edge('工经学院', '尧渡河路', line='4号线')
G.add_edge('尧渡河路', '杨柳塘', line='4号线')
G.add_edge('杨柳塘', '南艳湖', line='4号线')
G.add_edge('南艳湖', '九华山路', line='4号线')
G.add_edge('九华山路', '铜陵路', line='4号线')
G.add_edge('铜陵路', '九华南路', line='4号线')

# 添加5号线的边
G.add_edge('望湖城', '合肥南站', line='5号线')
G.add_edge('合肥南站', '盛大', line='5号线')
G.add_edge('盛大', '包河苑', line='5号线')
G.add_edge('包河苑', '义兴', line='5号线')
G.add_edge('义兴', '大连路', line='5号线')
G.add_edge('大连路', '花园大道', line='5号线')
G.add_edge('花园大道', '黄河路', line='5号线')
G.add_edge('黄河路', '扬子江路', line='5号线')
G.add_edge('扬子江路', '义城', line='5号线')
G.add_edge('义城', '省政务中心东', line='5号线')
G.add_edge('省政务中心东', '方兴湖', line='5号线')
G.add_edge('方兴湖', '渡江纪念馆', line='5号线')
G.add_edge('渡江纪念馆', '沈湾', line='5号线')
G.add_edge('沈湾', '华山路', line='5号线')
G.add_edge('华山路', '云谷路', line='5号线')
G.add_edge('云谷路', '清水冲', line='5号线')
G.add_edge('清水冲', '云川公园', line='5号线')
G.add_edge('云川公园', '滨湖竹园', line='5号线')
G.add_edge('滨湖竹园', '贵阳路', line='5号线')
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=3, node_color="skyblue", node_shape="o",alpha=0.8,width=2)
plt.show()

# # 定义每条地铁线路对应的颜色
# line_colors = {
#     '1号线': 'red',
#     '2号线': 'green',
#     '3号线': 'blue',
#     '4号线': 'purple',
#     '5号线': 'orange'
# }

# # 绘制每条地铁线路
# for line_name, color in line_colors.items():
#     # 获取当前地铁线路的所有站点
#     stations = [station for station in G.nodes if G.nodes[station].get('line') == line_name]
#     # 绘制当前地铁线路的边
#     for i in range(len(stations) - 1):
#         plt.plot([stations[i], stations[i+1]], [i, i+1], c=color, linewidth=2)
#         plt.show()
# # 显示地铁图
# plt.show()
# plt.savefig('subway_map.png')