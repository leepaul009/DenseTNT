


## 
```
instance = argoverse_get_instance(lines, file, args) #读取、处理csv数据
file_name: csv文件路径
start_time: 第一条数据的时间？可以理解为这个csv中的最初时间戳
city_name:
cent_x: AGENT在第20 time step的位置，agent在agoverse里是唯一的
cent_y:
agent_pred_index: 20
two_seconds: 所有agent在第20 time step的时间、减去开始时间（start_time）
angle:
traj: empty
agents: 每个agent的traj？shape=[20,2]
map_start_polyline_idx: ex 24, 之前的plyline_spans是agent traj的，之后是map的。
polygons: map中polygons，每个polygon存储着对应车道的中线位置，shape可以是[10, 2]
goals_2D: [num_goals, 2], goal point是在中心线上的两个相邻点之间采样的，确保采样点间距小于1.0，基本就还是在中心线上
goals_2D_label： 和AGENT的gt路径的最末点、距离最近的goal point在goals_2D上的序号
stage_one_label: AGENT的gt路径(shape=[30, 2])的最末点 和哪个polygon最近，记录该polygon的id（其实就是在这polygon上面）
matrix: 特征, shape=[num_nodes, 128]
labels: shape=[30, 2] AGENT的gt路径坐标
polyline_spans: 记录matrix上每个polyline上的起止位置，用来读取某个polyline的对应特征, 如feat_for_this_polyline, shape=[此polyline的node数量， 128]
labels_is_valid: shape=[30, ] 值是1或0
eval_time: 30


后续：
下面两个是在att处理gt所在polygon的过程中计算生成的：
stage_one_scores：[map_polyline_num] 记录每个polyline的分数
stage_one_topk：选取的topK的polyline的subgraph的map特征，[选取的topK的polyline, 128]

goals_2D_labels：类型是index。topk goal经过范围搜索后的candidates中，和AGENT goal最近的。


```


```
## stage 1：
用来学习“哪一个map polyline是gt match的车道”

#### Map的注意力特征
stage_one_cross_attention:
这里的输入都是subgraph特征。拿map特征作为query、traj+map总特征作为key和value做attention，这里面有head=1，head维度不参与att。
所以需要做一个维度转变：[batch, seq, feat*head] => [batch, head, seq, feat]，旨在seq维度做att。

#### 残差MLP
stage_one_decoder:
输入：globalgraph特征G，subgraph的map特征M、subgraph的map注意力特征AT
  cat(G, M, AT) => Feat;         注意: G是AGENT的feat，相当于seq=1，需要expand成[seq, feat], 目的是将AGENT的特征和每一个map polyline特征、注意力特征连接起来。
  cat(Feat, MLP(Feat)) => Feat;  注意: MLP的输入、输出维度是(128*3, 128)。 这里的残差比较特殊，是用concat，而不是element-wise add
  fc(Feat) => scores;            注意: fc的输入、输出维度是(128*4, 1)。输出是地图中每个polyline的分数

计算出以下几项：
  stage_one_topk: 按照score排序后的subgraph地图特征
  stage_one_scores: score


------------------------------------
## 用ATT计算goal score
1）输入goal的参考位置（中心线点的位置）和globalgraph的AGENT的特征，计算得到goal特征
2）把goal特征作为query，和subgraph的总特征做为key和value，做ATT得到goals_2D_hidden_attention
3）把goal特征作为query，topK的subgraph map特征作为key和value，做ATT得到stage_one_goals_2D_hidden_attention
将step1、2、3计算的得到的“三个特征”和AGENT的特征concat起来（在最后一个维度）得到一个特征，输入给decode得到每个goal的score


------------------------------------
## goals_2D_per_example_lazy_points 再用top-k goal，以ATT计算goal score
选取top-k个（150个）对应的goal points。
对于每个goal point选取一个grid（前后左右2.0范围）扩充为更多的goal candidates point。
用goal candidates作为输入，用“get_scores方法”计算得到score、和有最佳分数的best_goal。
在扩充过的goal candidates中，选取一个goal，它和当前AGENT gt goal最近。令goals_2D_labels=此goal candidates的index。

------------------------------------
取best_goal和AGENT GT goal做loss。
如果需要计算整个路径，则拿GT goal和AGENT计算target feat。
用target feat（query）和subgraph feat做att
用global feat、target feat和att feat一起decode得到traj，然后和gt traj一起计算smooth L1 loss。
用goals_2D_labels（离gt goal最近的candidates goal）和score（上一步）一起计算negative log likelihood loss。

```
