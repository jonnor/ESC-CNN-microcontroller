import pandas

sections = {
    'cover': 1,
    'summary': 1,
    'toc': 2,
    'introduction': 5,
    'background': 15,
    'materials': 6,
    'methods': 5,
    'results': 5,
    'discussion': 2,
    'conclusion': 1,
    'references': 2,
    'attachments': 5,
}
df = pandas.DataFrame({ 'pages': list(sections.values())}, index=list(sections.keys()))
print(df)

print(df.pages.sum())
