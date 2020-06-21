function cov_t = compute_cov(data)
  num_t=length(data);%统计训练图像集个数
  cov_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i});%取出每一个图像集,并中心化操作
  cov_temp=sample_t*sample_t';
  cov_t{i} = cov_temp+trace(cov_temp)*(1e-3)*eye(size(cov_temp,1));
   %从上面计算来看，线性子空间维数q=40，而论文中的D=400，这两个参数仅针对ETH-80数据集
end
end

