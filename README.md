# DSTCGCN

In this paper, we propose DSTCGCN, a GCN-based network to learn dynamic spatial and temporal dependencies jointly via graphs for traffic forecasting.

Generally, the main idea of this work lies in constructing dynamic cross graphs by fusing dynamic spatial graphs and dynamic temporal connection graphs in a directed and sparse way. 
- Dynamic spatial graphs are obtained by learnable parameters referred as node embeddings and time embeddings. In particular, "dynamic" for the spatial aspect is mainly realized by time embeddings representing hidden temporal features for each time step. 
- Dynamic temporal connection graphs are constructed based on an FFT-based attentive selector and dynamic spatial graphs. The concept of temporal connection graph comes from STSGCN and STFGCN at first. It is initially an identity matrix to enhance the self-connection for each node between adjacency time steps. Here, we design the FFT-based attentive selector to calculate the time-varying weights based on the real-time inputs, which can make the temporal connection graphs not only "dynamic" for weights but also "dynamic" for connected time steps.
- For the fusion part, considering the computational cost, we hope the fused graphs should be sparse. Thus, we introduce some acknowledged basis for design. For example, the past time steps can influence the future time steps, and the latest time step of the inputs should be the priority.

Full codes will be released ASAP
