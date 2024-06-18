import sys
import json


def main(args):
    model = args[2]
    metric = args[1]

    # Read JSON data from file
    with open(args[0], 'r') as file:
        data = json.load(file)

    # Extract test set data
    # test_data = data['test']
    test_data = data['test']

    metric_short = "MAE" if metric == "mean_absolute_error" else "MSE"

    # Define LaTeX table header
    header = """
    \\begin{table}[ht]
    \\centering
    \\begin{tabular}{|c|c|c|c|c|c|c|}
    \\hline
    """ + \
    "Model & N & Accuracy & \\multicolumn{3}{c|}{" + f"{metric_short}" + "}\\\\" + \
    """
    \\cline{4-6}
    & & & Range & Cost & Velocity \\\\
    \\hline
    """

    # Define LaTeX table footer
    footer = """
    \\hline
    \\end{tabular}
    \\caption{Model Performance on Test Set}
    \\label{tab:performance}
    \\end{table}
    """

    objectives = ['range', 'cost', 'velocity']

    table_entries = []
    for obj in objectives:

        obj_metric_mean = test_data[obj][metric]['mean']
        obj_metric_std = test_data[obj][metric]['std']

        table_entry = f"{obj_metric_mean:.2f} $\pm$ {obj_metric_std:.2f}"
        table_entries.append(table_entry)

    n = test_data['n'][0]
    accuracy = test_data['result']['accuracy_score']['mean']
    accuracy_std = test_data['result']['accuracy_score']['std']
    row = f"{model} & {n} & {accuracy:.2f} $\pm$ {accuracy_std:.2f} &" + " & ".join(table_entries) + "\\\\"

    # Extract required test set metrics

    # Combine header, row, and footer to form the complete table
    latex_table = header + row + footer

    # Print the LaTeX table
    print(latex_table)


if __name__ == "__main__":
    main(sys.argv[1:])