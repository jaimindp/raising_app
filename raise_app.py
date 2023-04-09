
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import plotly.graph_objs as go
import plotly.subplots as sp
import traceback


# st.set_page_config(layout='wide')
# st.config.get_option("server.enableCORS")
# st.set_option('deprecation.showPyplotGlobalUse', False)

params = st.experimental_get_query_params()


# Plot an investor on a chart
def investor_data_plotly(investor, upper_lim=None, lower_lim=None):
    
    investors = possible_matches[investor.lower()]
    
    investor_df = raises_clean[raises_clean.Investor.isin(investors)]
    
    if upper_lim:
        investor_df = investor_df[investor_df['Amount Raised'] < upper_lim]
    if lower_lim: 
        investor_df = investor_df[investor_df['Amount Raised'] > lower_lim]
        
    investor_group = investor_df.groupby(['Date', 'Company', 'Amount Raised'])

    times, names, amounts = [],[],[]
    for i in investor_group.groups:
        times.append(i[0])
        names.append(i[1])
        amounts.append(i[2])

    x, y, z = names, times, amounts
    together = [(names[i], times[i], amounts[i]) for i in range(len(x))]
    together.sort()

    text = [x for (x, y, z) in together]
    eucs = [y for (x, y, z) in together]
    covers = [z for (x, y, z) in together]

    fig = sp.make_subplots(specs=[[{"secondary_y": True}]])
    
#     fig.add_trace(go.Scatter(x=eucs, y=covers, mode='markers'), secondary_y=False)
    fig.add_trace(go.Scatter(x=eucs, y=covers, mode='markers', 
                             hovertemplate='%{text}<br>Raised: $%{y}<extra></extra>',
                             text=text), secondary_y=False)

    # Add annotations for each data point
    for t, x, y in zip(text, eucs, covers):
        fig.add_annotation(
            x=x,
            y=y,
            text=t,
            showarrow=False,
            font=dict(size=10),
            xshift=5,
            yshift=5,
        )
        
    return fig



# Plot a cloud of the vc from the description
def plot_vc_cloud(vc, min_companies=3):
    investors = possible_matches[vc.lower()]
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title('Wordcloud from investment descriptions', fontsize=20)
    
    try:
        cloud_df = raises_description[raises_description.Investor.isin(investors)]
        
        if len(cloud_df) > min_companies:

            comment_words = ''

            mystopwords = ['crypto', 'blockchain', 'protocol','none', 'network', 'company']
            stopwords = set(STOPWORDS)
            stopwords.update(set(mystopwords))

            # iterate through the csv file
            for val in cloud_df.Description:

                # typecaste each val to string
                val = str(val)

                # split the value
                tokens = val.split()

                # Converts each token into lowercase
                for i in range(len(tokens)):
                    tokens[i] = tokens[i].lower()

                comment_words += " ".join(tokens)+" "

            wordcloud = WordCloud(width = 800, height = 500,
                            background_color ='white',
                            stopwords = stopwords,
                            min_font_size = 10).generate(comment_words)


            ax.imshow(wordcloud)
            ax.axis("off")
            
            return ax, fig

    except Exception as e:
        print('Error', traceback.format_exc())
        

raises_clean = pd.read_csv('raises_clean.csv')
with open('possible_matches.json','r') as f:
    possible_matches = json.load(f)

raises_clean.Date = pd.to_datetime(raises_clean.Date)
raises_description = raises_clean[raises_clean.Description != 'None']

investor_names = raises_clean.Investor.unique()
st.subheader('Data from %d Raises' % len(raises_clean.groupby(['Company', 'Date','Amount Raised']).size()))

# Define the tab names
tabs = ["Specific investor lookup", "Find investors by sector"]

# Create a selectbox for tab selection
selected_tab = st.selectbox("Investor or sector", tabs)

# Display content based on the selected tab
if selected_tab == tabs[0]:
    
    investor = st.selectbox('Search for a specific investor', sorted(investor_names))

    fig = investor_data_plotly(investor)
    st.plotly_chart(fig)
    
    wordcloud = plot_vc_cloud(investor)
    
    if wordcloud:
        st.pyplot(wordcloud[1])

    st.subheader('Latest investments')
    specfic_investor_df = raises_clean[['Company','Investor','Date','Amount Raised']][raises_clean.Investor==investor].sort_values(by='Date', ascending=False).reset_index(drop=True).head(20)

    st.table(specfic_investor_df)


elif selected_tab == tabs[1]:

    # keyword = st.selectbox('Select Keyword', sorted(['cex','cefi','yield','exchange','custody','cybersecurity','dao','dex','defi','identity','game','gaming','hardware','healthcare','infrastructure','insurance','iot','l1','l2','mev','metaverse','mining','nft','news','payments','scaling','security','audits','social','storage','taxes','trading','web3', 'zk','aggregator']))
    c1, c2 = st.columns(2)

    keyword = c1.text_input('Keyword, e.g. defi, social, payments, nft etc.', value='defi')
    threshold = c2.number_input('Min investments',step=1, value=1)

    with open('keyword_lookup.json','r') as f:
        keywords_dict = json.load(f)

    def keyword_lookup(keyword, threshold=3, optimize='sector'):
        results = []
        for k,v in keywords_dict.items():
            if keyword in v['keywords']:
                results.append([k, v['keywords'][keyword]/v['Investments'] * 100, v['keywords'][keyword]])

        if optimize == 'sector':
            results.sort(key=lambda x: -x[1])
        else:
            results.sort(key=lambda x: -x[2])
        return list(filter(lambda x: x[2] >= threshold, results))

    keyword_df = pd.DataFrame(keyword_lookup(keyword,threshold), columns=['Name', '%% "%s" investments' % keyword, '# "%s" investments' % keyword]).sort_values(['# "%s" investments' % keyword], ascending=False).reset_index(drop=True)

    st.dataframe(keyword_df, height=800, use_container_width=True)


    # form = st.form(key="Form1")
    # c1, c2, c3, c4 = st.columns(4)

    # with c1:
    #     initialInvestment = form.text_input("Starting capital",value=500)
    # with c2:
    #     monthlyContribution = form.text_input("Monthly contribution (Optional)",value=100)
    # with c3:
    #     annualRate = form.text_input("Annual increase rate in percentage",value="15")
    # with c4:
    #     investingTimeYears = form.text_input("Years of investing:",value=10)

    # submitButton = form.form_submit_button(label = 'Calculate')
