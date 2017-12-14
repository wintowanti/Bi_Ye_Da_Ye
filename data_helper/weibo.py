#coding=UTF-8
from jieba import cut as weibo_tokenize
from data_helper.social_text import Social_Text
class WeiBo(Social_Text):
    def add_filter_word(self):
        raw_words = u" ，,的,。, ,了,是,#,不,我,？,在,/,、,都,就,也,有,你,“,”,和,人,：,:,…,说,要," \
                    u"还,一个,@,【,这,】,没有,吗,我们,会,就是,又,对,啊,t,被,让,cn,http,可以,给,现在,吧,上," \
                    u"自己,到,没,|,还是,看,但,去,很,能,什么,这个,过,为,不是,...,呢,来,着,想,-,大,把"
        for word in raw_words.split(u","):
            #print word
            self.filter_word.add(word)
            pass
    def __init__(self, raw_text, raw_target, stance, flag):
        self.tokenize_f = weibo_tokenize
        super(WeiBo, self).__init__(raw_text.decode("UTF-8"), raw_target.decode("UTF-8"), stance, flag)
        #self.add_filter_word()
