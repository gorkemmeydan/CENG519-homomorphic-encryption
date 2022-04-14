import os
import pandas as pd
import matplotlib.pyplot as plt

script_dir = os.path.dirname(__file__)
results_dir = os.path.join(script_dir, 'results/')

df = pd.read_csv('results.csv')
df.drop("SimCnt", axis=1)
# print(df)

by_node_count = df.groupby(by="NodeCount")
# for key, item in by_node_count:
#     print(by_node_count.get_group(key))

average = by_node_count.mean()
std = by_node_count.std()
# print(average)

########
node_simcnt_avg = average[["CompileTime"]]
node_simcnt_std = std[["CompileTime"]]
p = node_simcnt_avg.plot(kind="bar", yerr=node_simcnt_std)
p.set_title("Node Count vs Compile Time")
p.set_xlabel("Node Count")
p.set_ylabel("Compile Time (ms)")
plt.savefig(results_dir + "node_vs_compiletime")

node_keygen_avg = average[["KeyGenerationTime"]]
node_keygen_std = std[["KeyGenerationTime"]]
p = node_keygen_avg.plot(kind="bar", yerr=node_keygen_std)
p.set_title("Node Count vs Key Generation Time")
p.set_xlabel("Node Count")
p.set_ylabel("Key Generation Time (ms)")
plt.savefig(results_dir + "node_vs_keygentime")

node_enc_avg = average[["EncryptionTime"]]
node_enc_std = std[["EncryptionTime"]]
p = node_enc_avg.plot(kind="bar", yerr=node_enc_std)
p.set_title("Node Count vs Encryption Time")
p.set_xlabel("Node Count")
p.set_ylabel("Encryption Time (ms)")
plt.savefig(results_dir + "node_vs_encrpytiontime")

node_exec_avg = average[["ExecutionTime"]]
node_exec_std = std[["ExecutionTime"]]
p = node_exec_avg.plot(kind="bar", yerr=node_exec_std)
p.set_title("Node Count vs Execution Time")
p.set_xlabel("Node Count")
p.set_ylabel("Execution Time (ms)")
plt.savefig(results_dir + "node_vs_executiontime")

node_dec_avg = average[["DecryptionTime"]]
node_dec_std = std[["DecryptionTime"]]
p = node_dec_avg.plot(kind="bar", yerr=node_dec_std)
p.set_title("Node Count vs Decryption Time")
p.set_xlabel("Node Count")
p.set_ylabel("Decryption Time (ms)")
plt.savefig(results_dir + "node_vs_decryptiontime")

node_ref_avg = average[["ReferenceExecutionTime"]]
node_ref_std = std[["ReferenceExecutionTime"]]
p = node_ref_avg.plot(kind="bar", yerr=node_ref_std)
p.set_title("Node Count vs Reference Execution Time")
p.set_xlabel("Node Count")
p.set_ylabel("Reference Execution Time (ms)")
plt.savefig(results_dir + "node_vs_referenceexecutiontime")

node_mse_avg = average[["Mse"]]
node_mse_std = std[["Mse"]]
p = node_mse_avg.plot(kind="bar", yerr=node_mse_std)
p.set_title("Node Count vs MSE")
p.set_xlabel("Node Count")
p.set_ylabel("MSE")
plt.savefig(results_dir + "node_vs_mse")
