function cov_t = compute_cov(data)
  num_t=length(data);%ͳ��ѵ��ͼ�񼯸���
  cov_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i});%ȡ��ÿһ��ͼ��,�����Ļ�����
  cov_temp=sample_t*sample_t';
  cov_t{i} = cov_temp+trace(cov_temp)*(1e-3)*eye(size(cov_temp,1));
   %��������������������ӿռ�ά��q=40���������е�D=400�����������������ETH-80���ݼ�
end
end

