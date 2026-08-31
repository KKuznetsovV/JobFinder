from jobfinder.db.models import ApplyMethod
from jobfinder.sources.devjobs import DevJobsSource, _parse_listing_page

# Trimmed but structurally real markup, captured from a live render of
# devjobs.co.il/jobs-grid on 2026-08-31.
SAMPLE_HTML = """
<html><body>
<div class="col-xl-4"><div class="card-grid-2 hover-up newDesign">
  <div class="card-grid-2-image-left">
    <div class="right-info">
      <a class="profession" href="https://devjobs.co.il/company-details/nvidia">NVIDIA</a>
      <a class="name-job" target="_blank" href="https://devjobs.co.il/job-details/4459744883">IC Digital Test Engineer</a>
      <div><span class="location-small">Yokneam Ilit (Hybrid)</span><span class="card-time">Aug 31, 2026</span></div>
    </div>
  </div>
  <div class="card-block-info">
    <ul class="courses">
      <li class="btn btn-grey-small">Python</li>
      <li class="btn btn-grey-small">Perl</li>
    </ul>
    <div class="card-2-bottom">
      <a class="btn btn-apply-now login-popup" data-url="https://il.linkedin.com/jobs/view/4459744883" data-jobid="4459744883">Apply now</a>
    </div>
  </div>
</div></div>
<div class="col-xl-4"><div class="card-grid-2 hover-up newDesign">
  <div class="card-grid-2-image-left">
    <div class="right-info">
      <a class="profession" href="https://devjobs.co.il/company-details/driivz">Driivz</a>
      <a class="name-job" target="_blank" href="https://devjobs.co.il/job-details/4451316670">Backend Team Lead</a>
      <div><span class="location-small">Israel (On-site)</span><span class="card-time">Aug 30, 2026</span></div>
    </div>
  </div>
  <div class="card-block-info">
    <ul class="courses">
      <li class="btn btn-grey-small">Java</li>
      <li class="btn btn-grey-small">Kotlin</li>
    </ul>
    <div class="card-2-bottom">
      <a class="btn btn-apply-now login-popup" data-url="https://careers.driivz.com/apply/4451316670" data-jobid="4451316670">Apply now</a>
    </div>
  </div>
</div></div>
</body></html>
"""


def test_parse_listing_page_extracts_cards():
    jobs = _parse_listing_page(SAMPLE_HTML)

    assert len(jobs) == 2
    first = jobs[0]
    assert first["external_id"] == "4459744883"
    assert first["title"] == "IC Digital Test Engineer"
    assert first["company"] == "NVIDIA"
    assert first["location"] == "Yokneam Ilit (Hybrid)"
    assert first["posted_date"] == "Aug 31, 2026"
    assert first["skills"] == ["Python", "Perl"]
    assert first["apply_url"] == "https://il.linkedin.com/jobs/view/4459744883"


def test_devjobs_source_sets_apply_method_from_apply_url(mocker):
    mocker.patch(
        "jobfinder.sources.devjobs.scrape_utils.render_page_html", return_value=SAMPLE_HTML
    )

    postings = DevJobsSource().fetch()

    assert len(postings) == 2
    assert postings[0].apply_method == ApplyMethod.LINKEDIN_EASY_APPLY
    assert postings[0].url == "https://il.linkedin.com/jobs/view/4459744883"
    assert postings[1].apply_method == ApplyMethod.COMPANY_SITE
    assert postings[1].url == "https://careers.driivz.com/apply/4451316670"
    assert "Skills: Python, Perl" in postings[0].description


def test_devjobs_source_returns_empty_on_render_failure(mocker):
    mocker.patch(
        "jobfinder.sources.devjobs.scrape_utils.render_page_html",
        side_effect=RuntimeError("boom"),
    )

    assert DevJobsSource().fetch() == []
