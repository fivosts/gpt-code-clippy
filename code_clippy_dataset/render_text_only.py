from mistune import Renderer


class TextRenderer(Renderer):
    """
    A text-only renderer. Useful from stripping markdown from a document
    https://raw.githubusercontent.com/lepture/mistune-contrib/601dafcd3486f6696591271f48b7b8e1b2802335/mistune_contrib/render_text_only.py
    """
    def block_code(self, code, lang=None):
        return code+"\n"

    def block_quote(self, text):
        return text+"\n"

    def block_html(self, html):
        return '\n'

    def header(self, text, level, raw=None):
        return text+"\n"

    def hrule(self):
        return '\n'

    def list(self, body, ordered=True):
        return body+"\n"

    def list_item(self, text):
        return text+"\n"

    def paragraph(self, text):
        return ("%s\n" % text.strip(' ')) + "\n"

    def table(self, header, body):
        return ('%s\n%s' % (header, body)) + "\n"

    def table_row(self, content):
        return content+"\n"

    def table_cell(self, content, **flags):
        return " %s " % content

    def double_emphasis(self, text):
        return text

    def emphasis(self, text):
        return text

    def codespan(self, text):
        return text

    def linebreak(self):
        return '\n'

    def strikethrough(self, text):
        return text

    def text(self, text):
        return text

    def autolink(self, link, is_email=False):
        return ''

    def link(self, link, title, text):
        return text

    def image(self, src, title, text):
        return ("%s %s" % (title, text)) + "\n"

    def inline_html(self, html):
        return ''

    def footnote_ref(self, key, index):
        return ''

    def footnote_item(self, key, text):
        return text

    def footnotes(self, text):
        return text
