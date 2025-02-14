import numpy as np
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from tensorflow.keras.models import Model # type: ignore

# 模拟数据：用户ID、广告ID和点击情况
user_ids = np.array([1, 2, 3, 4, 5])
ad_ids = np.array([1, 2, 3, 4, 5])
clicks = np.array([1, 0, 0, 1, 1])  # 1表示点击，0表示没有点击

# 定义输入层：用户ID和广告ID
user_input = layers.Input(shape=(1,), dtype=tf.int32, name="user_id")
ad_input = layers.Input(shape=(1,), dtype=tf.int32, name="ad_id")

# 创建嵌入层（Embedding Layer）：为用户和广告ID创建低维向量表示
user_embedding = layers.Embedding(input_dim=6, output_dim=4)(user_input)  # 6是用户的总数，4是嵌入维度
ad_embedding = layers.Embedding(input_dim=6, output_dim=4)(ad_input)  # 6是广告的总数，4是嵌入维度

# 扁平化嵌入层的输出
user_vec = layers.Flatten()(user_embedding)
ad_vec = layers.Flatten()(ad_embedding)

# 合并用户向量和广告向量
merged = layers.concatenate([user_vec, ad_vec])

# 添加全连接层
x = layers.Dense(8, activation='relu')(merged)
output = layers.Dense(1, activation='sigmoid')(x)  # 使用sigmoid激活函数来预测点击概率

# 构建和编译模型
model = Model(inputs=[user_input, ad_input], outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 模型概况
model.summary()

# 训练模型
model.fit([user_ids, ad_ids], clicks, epochs=10, batch_size=1)

# 模拟一个新的用户和广告对，预测是否点击
new_user_id = np.array([1])
new_ad_id = np.array([2])
prediction = model.predict([new_user_id, new_ad_id])

print(f"预测点击概率：{prediction[0][0]:.4f}")
