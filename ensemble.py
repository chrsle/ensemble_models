{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "lee_chris_Homework 7.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "O38dWgFrTt9g"
      },
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import MinMaxScaler\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import BaggingClassifier\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.ensemble import AdaBoostClassifier\n",
        "from sklearn.ensemble import VotingClassifier\n",
        "from sklearn.dummy import DummyClassifier\n",
        "from sklearn.tree import DecisionTreeClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "from sklearn.svm import SVC\n",
        "from sklearn.linear_model import LogisticRegression"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "U2II5L3AXn2A"
      },
      "source": [
        "pd.set_option('display.max_columns', 100)\n",
        "data = pd.read_csv('cancer.csv')"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3bbOI9AYTziJ"
      },
      "source": [
        "data.drop(['Sample Code Number'],axis = 1, inplace = True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cHu_DIDcZs_8",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e3c43c91-e3dc-43b3-8dcc-8aefd7dd3d2d"
      },
      "source": [
        "data['Bare Nuclei']"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0       3\n",
              "1       3\n",
              "2       3\n",
              "3       3\n",
              "4       3\n",
              "       ..\n",
              "694     1\n",
              "695     1\n",
              "696     8\n",
              "697    10\n",
              "698    10\n",
              "Name: Bare Nuclei, Length: 699, dtype: int64"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 5
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "u_5GSsVjU930"
      },
      "source": [
        "data.replace('?',0, inplace=True)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "LV3PVQB0U_mF"
      },
      "source": [
        "# Convert the DataFrame object into NumPy array otherwise you will not be able to impute\n",
        "values = data.values\n",
        "\n",
        "# Now impute it\n",
        "imputer = SimpleImputer(missing_values=np.nan, strategy='mean')\n",
        "imputedData = imputer.fit_transform(values)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "VomHojxET85c",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "5eecd9fc-5d66-49ef-8b09-7bbc8254c97b"
      },
      "source": [
        "scaler = MinMaxScaler(feature_range=(0, 1))\n",
        "normalizedData = scaler.fit_transform(imputedData)\n",
        "cols = ['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bland Chromatin', 'Bare Nuclei', 'Normal Nucleoli', 'Mitosis','Class']\n",
        "normalizedData = pd.DataFrame(normalizedData, columns=cols)\n",
        "print(normalizedData.head())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "   Clump Thickness  Uniformity of Cell Size  Uniformity of Cell Shape  \\\n",
            "0         0.444444                 0.000000                  0.000000   \n",
            "1         0.444444                 0.333333                  0.333333   \n",
            "2         0.222222                 0.000000                  0.000000   \n",
            "3         0.555556                 0.777778                  0.777778   \n",
            "4         0.333333                 0.000000                  0.000000   \n",
            "\n",
            "   Marginal Adhesion  Single Epithelial Cell Size  Bland Chromatin  \\\n",
            "0           0.000000                     0.111111              0.1   \n",
            "1           0.444444                     0.666667              1.0   \n",
            "2           0.000000                     0.111111              0.2   \n",
            "3           0.000000                     0.222222              0.4   \n",
            "4           0.222222                     0.111111              0.1   \n",
            "\n",
            "   Bare Nuclei  Normal Nucleoli  Mitosis  Class  \n",
            "0     0.222222         0.000000      0.0    0.0  \n",
            "1     0.222222         0.111111      0.0    0.0  \n",
            "2     0.222222         0.000000      0.0    0.0  \n",
            "3     0.222222         0.666667      0.0    0.0  \n",
            "4     0.222222         0.000000      0.0    0.0  \n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gOshE9K_UA5S",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "8434a713-e5f7-4f9e-9379-b15c815bf97b"
      },
      "source": [
        "X = normalizedData.iloc[:,0:9]\n",
        "y = normalizedData.iloc[:,9]\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)\n",
        "d = DummyClassifier(strategy='most_frequent')\n",
        "d.fit(X_train, y_train)\n",
        "d.score(X_test, y_test)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "0.680952380952381"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "C78-znrIUDcs",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "12536816-3646-4be3-beb0-2833c815d970"
      },
      "source": [
        "# Generic Bagging model\n",
        "\n",
        "bag_model = BaggingClassifier(base_estimator = DecisionTreeClassifier(), n_estimators=10, random_state=42)\n",
        "bag_model = bag_model.fit(X_train, y_train)\n",
        "bag_pred = bag_model.predict(X_test)\n",
        "print(\"Generic Bagging Accuracy: {}\".format(accuracy_score(y_test, bag_pred)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Generic Bagging Accuracy: 0.9571428571428572\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TagawD02hKlF",
        "outputId": "68102868-2014-4d8b-ba20-5fee41b66c90"
      },
      "source": [
        "# Random Forest model\n",
        "\n",
        "rf_model = RandomForestClassifier(n_estimators=100, max_features=7, random_state=42)\n",
        "rf_model = rf_model.fit(X_train, y_train)\n",
        "rf_pred = rf_model.predict(X_test)\n",
        "print(\"Random Forest Accuracy: {}\".format(accuracy_score(y_test, rf_pred)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Random Forest Accuracy: 0.9619047619047619\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "M469DqsQxlFN",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 328
        },
        "outputId": "a9dcff6a-04a2-495c-f264-201e15300089"
      },
      "source": [
        "# Top 3 features for RandomForest\n",
        "\n",
        "imp = pd.DataFrame(zip(X_train.columns, rf_model.feature_importances_))\n",
        "imp.sort_values(by=[1], ascending=False)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Uniformity of Cell Size</td>\n",
              "      <td>0.396144</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Uniformity of Cell Shape</td>\n",
              "      <td>0.249673</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Bland Chromatin</td>\n",
              "      <td>0.176869</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Bare Nuclei</td>\n",
              "      <td>0.067437</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Clump Thickness</td>\n",
              "      <td>0.036890</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Normal Nucleoli</td>\n",
              "      <td>0.029999</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Single Epithelial Cell Size</td>\n",
              "      <td>0.022517</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Marginal Adhesion</td>\n",
              "      <td>0.016603</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Mitosis</td>\n",
              "      <td>0.003868</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                             0         1\n",
              "1      Uniformity of Cell Size  0.396144\n",
              "2     Uniformity of Cell Shape  0.249673\n",
              "5              Bland Chromatin  0.176869\n",
              "6                  Bare Nuclei  0.067437\n",
              "0              Clump Thickness  0.036890\n",
              "7              Normal Nucleoli  0.029999\n",
              "4  Single Epithelial Cell Size  0.022517\n",
              "3            Marginal Adhesion  0.016603\n",
              "8                      Mitosis  0.003868"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 12
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "DQIzaUjWUHv7",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "3f4f08cb-46b0-4c7e-d4c1-0bcb113d25ab"
      },
      "source": [
        "# AdaBoost Classification\n",
        "\n",
        "boost_model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=4), n_estimators=200, random_state=42, learning_rate =0.05)\n",
        "boost_model = boost_model.fit(X_train, y_train)\n",
        "boost_pred = boost_model.predict(X_test)\n",
        "print(\"AdaBoost Classification Accuracy: {}\".format(accuracy_score(y_test, boost_pred)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "AdaBoost Classification Accuracy: 0.9619047619047619\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jbEUjwQRw3rW",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 328
        },
        "outputId": "2a9925d0-437d-4c66-cc24-27e924593583"
      },
      "source": [
        "# Top 3 features for AdaBoost\n",
        "\n",
        "imp = pd.DataFrame(zip(X_train.columns, boost_model.feature_importances_))\n",
        "imp.sort_values(by=[1], ascending=False)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/html": [
              "<div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>0</th>\n",
              "      <th>1</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>Clump Thickness</td>\n",
              "      <td>0.194414</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>6</th>\n",
              "      <td>Bare Nuclei</td>\n",
              "      <td>0.190095</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>5</th>\n",
              "      <td>Bland Chromatin</td>\n",
              "      <td>0.142675</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>Uniformity of Cell Shape</td>\n",
              "      <td>0.131433</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>7</th>\n",
              "      <td>Normal Nucleoli</td>\n",
              "      <td>0.095740</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>Single Epithelial Cell Size</td>\n",
              "      <td>0.092518</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>Marginal Adhesion</td>\n",
              "      <td>0.055472</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>Uniformity of Cell Size</td>\n",
              "      <td>0.053106</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>8</th>\n",
              "      <td>Mitosis</td>\n",
              "      <td>0.044548</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>"
            ],
            "text/plain": [
              "                             0         1\n",
              "0              Clump Thickness  0.194414\n",
              "6                  Bare Nuclei  0.190095\n",
              "5              Bland Chromatin  0.142675\n",
              "2     Uniformity of Cell Shape  0.131433\n",
              "7              Normal Nucleoli  0.095740\n",
              "4  Single Epithelial Cell Size  0.092518\n",
              "3            Marginal Adhesion  0.055472\n",
              "1      Uniformity of Cell Size  0.053106\n",
              "8                      Mitosis  0.044548"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 14
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "3M3ZIcrqUKjK",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "853eb119-704b-4d62-95a8-524bddca908c"
      },
      "source": [
        "# Voting Ensemble for Classification\n",
        "\n",
        "rf = RandomForestClassifier(n_estimators=200)\n",
        "dt = DecisionTreeClassifier(max_depth=4)\n",
        "svm = SVC(probability=True)\n",
        "lr = LogisticRegression()\n",
        "\n",
        "eclf = VotingClassifier(estimators=[('rf', rf), ('dt', dt), ('svm', svm), ('lr', lr)], voting='soft')\n",
        "eclf = eclf.fit(X_train, y_train)\n",
        "eclf_pred = eclf.predict(X_test)\n",
        "print(\"Ensemble Classification Accuracy: {}\".format(accuracy_score(y_test, eclf_pred)))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Ensemble Classification Accuracy: 0.9666666666666667\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "gj09m4z_lXNB"
      },
      "source": [
        "#Ensemble"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}