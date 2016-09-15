import datetime
from bravado.client import SwaggerClient
from bravado.requests_client import RequestsClient
from bravado.fido_client import FidoClient


class NeptuneScripter(object):
    def __init__(self):
        config = {
            # validate the swagger spec
            'validate_swagger_spec': False,
            'models': True,
        }
        self.client = SwaggerClient.from_url("http://173.192.46.42:8080/static/swagger.json", config=config)
        self.client.swagger_spec.api_url = 'http://173.192.46.42:8080/0.6/default/'
        print self.client.swagger_spec.api_url

    def mark_running_aborted(self):
        ids = map(lambda job: job.id, self.client.jobs.get_jobs(tags=['maciek']).result())

        job_futures = map(lambda id: self.client.jobs.get_jobs_jobId(jobId=id), ids)
        print dir(self.client.jobs)
        print ids
        for job_future in job_futures:
            job = job_future.result()
            print job.id
            if job.state == 'running':
                update = {
                    "completionStatus": "succeeded",
                    "traceback": "",
                    "timeOfCompletion": "2016-07-04T14:10:57.282Z"
                }
                print self.client.jobs.post_jobs_jobId_markCompleted(jobId=job.id, completedJobParams=update).result()

    def find_jobs(self):
        for job in self.client.jobs.get_jobs(tags=['maciek', 'rl']).result():
            print job.state, job.substate


    def find_spam(self):
        def is_spam(job):
            properties = job.properties
            state = job.state

            print job.channelsLastValues
            #print properties, state, timeOfCreation, timeOfCompletion

            epochs_done_thresh = 15

            if 'wazne' in job.tags:
                return False

            if state == 'completed':
                datatimeOfCompletion = job.timeOfCompletion
                datatimeOfCompletion = datatimeOfCompletion.replace(tzinfo=None)
                print datatimeOfCompletion, 'fdfdfd'
                #print type(datatimeOfCompletion)
                now = datetime.datetime.now()
                naive_now = now.replace(tzinfo=None)
                #print type(now)
                diff = naive_now - datatimeOfCompletion
                print naive_now, datatimeOfCompletion
                print diff.seconds, 'secohnds'

                epochs_done = None

                for a in job.channelsLastValues:
                    print a.channelName
                    if a.channelName == 'epochs_done':
                        epochs_done = float(a.y)
                print epochs_done

                if epochs_done is not None:
                    if epochs_done < epochs_done_thresh and diff.seconds >= 120 * 60:
                        return True
                    else:
                        return False
                else:
                    return True

            else:
                return False


        ids = map(lambda job: job.id, self.client.jobs.get_jobs(tags=['maciek']).result())
        job_futures = map(lambda id: self.client.jobs.get_jobs_jobId(jobId=id), ids)

        for job_future in job_futures:
            job = job_future.result()
        #    is_spam(job)
            print job.tags
            new_tags = job.tags
            # try:
            #     new_tags.index('spam')
            #     continue
            # except ValueError:
            #     pass

            spam = is_spam(job)
            print 'spam', id, spam
            if spam:
                if 'not_spam' in new_tags:
                    new_tags.remove('not_spam')
                if 'spam' in new_tags:
                    new_tags.remove('spam')
                new_tags.append('spam')
            else:
                if 'not_spam' in new_tags:
                    new_tags.remove('not_spam')
                if 'spam' in new_tags:
                    new_tags.remove('spam')
                new_tags.append('not_spam')

            update = {'tags': new_tags}
            print 'setting as spam', job.id
            self.client.jobs.put_jobs_jobId(jobId=job.id, name=update).result()

    def main(self):
        self.find_jobs()
        #self.find_spam()
        #self.mark_running_aborted()


def main():
    scripter = NeptuneScripter()
    scripter.main()

if __name__ == '__main__':
    main()
