function cov_t = compute_cov(data)
  num_t=length(data);
  cov_t=cell(1,num_t);
for i=1:num_t
  sample_t=zscore(data{i});
  cov_temp=sample_t*sample_t';
  cov_t{i} = cov_temp+trace(cov_temp)*(1e-3)*eye(size(cov_temp,1));
end
end

