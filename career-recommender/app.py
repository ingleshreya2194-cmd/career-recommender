"""Career & Course Recommender - SDG 4 (Quality Education) & SDG 8 (Decent Work)."""
import numpy as np
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

SKILLS = ["Programming", "Maths", "Communication", "Creativity", "Problem solving", "Business sense"]
INTERESTS = ["Data & AI", "Web & apps", "Design", "Security & cloud", "Business & marketing", "Teaching"]

# skills(6), interests(6), typical marks
CAREERS = {
    "Data Scientist": dict(
        p=[7, 9, 5, 5, 8, 4], i=[1, 0, 0, 0, 0, 0], m=75,
        about="Turn raw data into predictions and decisions using statistics and machine learning.",
        path=["Python + Pandas + SQL", "Statistics and ML basics", "2-3 Kaggle / real-data projects", "Apply for data analyst internships"],
        courses=[("Kaggle Learn - Python, Pandas, ML", "https://www.kaggle.com/learn"),
                 ("fast.ai - Practical Deep Learning", "https://course.fast.ai"),
                 ("IBM SkillsBuild - Data Science", "https://skillsbuild.org")]),
    "Full Stack Developer": dict(
        p=[9, 6, 5, 6, 8, 3], i=[0, 1, 0, 0, 0, 0], m=65,
        about="Build complete web apps - the screens users see and the servers behind them.",
        path=["HTML, CSS, JavaScript", "React + Flask/Node", "Database + deployment", "Portfolio with 3 live projects"],
        courses=[("freeCodeCamp - Responsive Web + JS", "https://www.freecodecamp.org"),
                 ("The Odin Project - Full Stack", "https://www.theodinproject.com"),
                 ("CS50 - Intro to Computer Science", "https://cs50.harvard.edu")]),
    "UI/UX Designer": dict(
        p=[4, 3, 7, 10, 6, 5], i=[0, 0, 1, 0, 0, 0], m=60,
        about="Design apps and websites that are easy, clear and enjoyable to use.",
        path=["Design basics: colour, type, layout", "Learn Figma", "Redesign 3 existing apps", "Build a case-study portfolio"],
        courses=[("Google UX Design (financial aid on Coursera)", "https://www.coursera.org"),
                 ("Figma Learn - Design Basics", "https://help.figma.com/hc/en-us/categories/360002051613"),
                 ("Khan Academy - Computing & Design", "https://www.khanacademy.org")]),
    "Cybersecurity Analyst": dict(
        p=[8, 7, 5, 4, 9, 3], i=[0, 0, 0, 1, 0, 0], m=70,
        about="Protect systems and data from attacks by finding weaknesses before hackers do.",
        path=["Networking + Linux basics", "Security fundamentals", "Practice labs and CTFs", "Entry-level SOC / analyst roles"],
        courses=[("Cisco Networking Academy - Cybersecurity", "https://www.netacad.com"),
                 ("IBM SkillsBuild - Cybersecurity", "https://skillsbuild.org"),
                 ("NPTEL - Computer Networks & Security", "https://nptel.ac.in")]),
    "Digital Marketer": dict(
        p=[3, 3, 9, 8, 5, 8], i=[0, 0, 0, 0, 1, 0], m=60,
        about="Grow brands online through content, social media, SEO and ads.",
        path=["Marketing fundamentals", "SEO + social media", "Run a small real campaign", "Analytics and reporting"],
        courses=[("Google Digital Garage - Fundamentals", "https://learndigital.withgoogle.com/digitalgarage"),
                 ("HubSpot Academy - Digital Marketing", "https://academy.hubspot.com"),
                 ("SWAYAM - Marketing Management", "https://swayam.gov.in")]),
    "Cloud / DevOps Engineer": dict(
        p=[8, 6, 5, 3, 8, 4], i=[0, 1, 0, 1, 0, 0], m=65,
        about="Run applications reliably on the cloud with automation and monitoring.",
        path=["Linux + Git", "Docker and CI/CD", "One cloud platform (AWS/Azure)", "Deploy a project end to end"],
        courses=[("IBM SkillsBuild - Cloud Computing", "https://skillsbuild.org"),
                 ("NPTEL - Cloud Computing", "https://nptel.ac.in"),
                 ("freeCodeCamp - DevOps & Cloud videos", "https://www.freecodecamp.org")]),
    "Teacher / Ed-tech Educator": dict(
        p=[4, 5, 10, 7, 6, 4], i=[0, 0, 0, 0, 0, 1], m=70,
        about="Explain ideas clearly and help others learn, in classrooms or online.",
        path=["Master one subject deeply", "Practise explaining in simple words", "Create tutorials or videos", "Tutor, then apply to ed-tech / teaching roles"],
        courses=[("Khan Academy - Teaching resources", "https://www.khanacademy.org"),
                 ("SWAYAM - Education & Pedagogy", "https://swayam.gov.in"),
                 ("BBC Learning English - Communication", "https://www.bbc.co.uk/learningenglish")]),
    "Business Analyst": dict(
        p=[4, 7, 8, 5, 8, 9], i=[1, 0, 0, 0, 1, 0], m=70,
        about="Study business problems with data and recommend practical improvements.",
        path=["Excel + SQL", "Power BI / Tableau", "Requirement gathering and documentation", "Analyse a real business dataset"],
        courses=[("Google Digital Garage - Data & Analytics", "https://learndigital.withgoogle.com/digitalgarage"),
                 ("Kaggle Learn - SQL & Data Viz", "https://www.kaggle.com/learn"),
                 ("NPTEL - Business Analytics", "https://nptel.ac.in")]),
}
NAMES = list(CAREERS)


def train():
    rng = np.random.default_rng(42)
    X, y = [], []
    for idx, name in enumerate(NAMES):
        c = CAREERS[name]
        for _ in range(350):
            sk = np.clip(np.round(np.array(c["p"]) + rng.normal(0, 1.6, 6)), 1, 10)
            it = (rng.random(6) < np.where(np.array(c["i"]) == 1, 0.85, 0.12)).astype(float)
            mk = np.clip(rng.normal(c["m"], 12), 35, 100)
            X.append(list(sk) + list(it) + [mk])
            y.append(idx)
    model = RandomForestClassifier(n_estimators=120, random_state=42, n_jobs=1)
    model.fit(np.array(X), np.array(y))
    return model


MODEL = train()


@app.route("/")
def home():
    return render_template("index.html", skills=SKILLS, interests=INTERESTS)


@app.route("/health")
def health():
    return "ok"


@app.route("/api/recommend", methods=["POST"])
def recommend():
    d = request.get_json(force=True, silent=True) or {}
    try:
        sk = [min(10, max(1, int(d["skills"][k]))) for k in SKILLS]
        it = [1 if k in d.get("interests", []) else 0 for k in INTERESTS]
        marks = min(100, max(0, float(d.get("marks", 60))))
    except (KeyError, ValueError, TypeError):
        return jsonify(error="Please fill in all fields correctly."), 400

    proba = MODEL.predict_proba(np.array([sk + it + [marks]]))[0]
    top = np.argsort(proba)[::-1][:3]
    total = proba[top].sum() or 1
    out = []
    for i in top:
        c = CAREERS[NAMES[i]]
        gaps = sorted(range(6), key=lambda j: sk[j] - c["p"][j])[:2]
        strong = sorted(range(6), key=lambda j: sk[j] - c["p"][j], reverse=True)[:2]
        out.append(dict(
            name=NAMES[i], match=round(float(proba[i] / total) * 100), about=c["about"], path=c["path"],
            courses=[dict(title=t, url=u) for t, u in c["courses"]],
            strengths=[SKILLS[j] for j in strong if sk[j] >= c["p"][j] - 1],
            improve=[SKILLS[j] for j in gaps if sk[j] < c["p"][j] - 1]))
    return jsonify(results=out)


if __name__ == "__main__":
    app.run(debug=True)
