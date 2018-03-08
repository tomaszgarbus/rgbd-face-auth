#include <string>

#include <wx/wx.h>
#include <wx/panel.h>
#include <wx/image.h>
#include <wx/slider.h>
#include <wx/wxprec.h>
#include <wx/bitmap.h>
#include <wx/stattext.h>

#include "picture.hpp"

enum {
   ID_MIN_D = 101,
   ID_MAX_D = 102,
   ID_MIN_D_TEXT = 103,
   ID_MAX_D_TEXT = 104
};

class DisplayPanel;

class SettingsPanel : public wxPanel {
   DisplayPanel *picture_panel;

   wxPanel *m_parent;
   wxSlider *m_min_d, *m_max_d;
   wxTextCtrl *m_min_d_text, *m_max_d_text;

 public:
   SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel);

   void on_change(wxCommandEvent &event);
};

class DisplayPanel : public wxPanel {
   Picture *const picture;
   wxStaticBitmap *m_picture;

 public:
   DisplayPanel(wxPanel *parent, Picture *picture);

   void update_picture(int64_t const min_depth, int64_t const max_depth);
};

class MainWindow : public wxFrame {
   Picture *picture;
   wxPanel *m_parent;

   DisplayPanel *m_display;
   SettingsPanel *m_settings;

 public:
   MainWindow(const wxString& title);
};

// Definitions

SettingsPanel::SettingsPanel(wxPanel *parent, DisplayPanel *picture_panel)
      : wxPanel(parent, -1, wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN),
        picture_panel(picture_panel),
        m_parent(parent),
        m_min_d(new wxSlider(this, ID_MIN_D, 300, 0, 10000, wxPoint(60, 10), wxSize(980, 15))),
        m_max_d(new wxSlider(this, ID_MAX_D, 1000, 0, 10000, wxPoint(60, 40), wxSize(980, 15))),
        m_min_d_text(new wxTextCtrl(this, ID_MIN_D_TEXT, "300", wxPoint(10, 10), wxSize(40, 15))),
        m_max_d_text(new wxTextCtrl(this, ID_MAX_D_TEXT, "1000", wxPoint(10, 40), wxSize(40, 15))) {
   Connect(ID_MIN_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
   Connect(ID_MAX_D, wxEVT_SCROLL_CHANGED, wxCommandEventHandler(SettingsPanel::on_change));
}

void SettingsPanel::on_change(wxCommandEvent &event) {
   int64_t const min_depth = m_min_d->GetValue();
   int64_t const max_depth = m_max_d->GetValue();

   m_min_d_text->Clear();
   m_min_d_text->WriteText(std::to_string(m_min_d->GetValue()));
   m_max_d_text->Clear();
   m_max_d_text->WriteText(std::to_string(m_max_d->GetValue()));
   // TODO: Make wxTextCtrls non-editable or update values on edit.

   picture_panel->update_picture(min_depth, max_depth);
}

DisplayPanel::DisplayPanel(wxPanel *parent, Picture *picture)
      : wxPanel(parent, wxID_ANY, wxDefaultPosition, wxSize(640, 480), wxBORDER_SUNKEN),
        picture(picture) {
   wxImage tmp(640, 480);
   if (picture->color_frame != nullptr) {
      tmp.SetData(reinterpret_cast<uint8_t *>(picture->color_frame->pixels->data()));
   }

   delete m_picture;
   m_picture = new wxStaticBitmap(this, wxID_ANY, wxBitmap(tmp), wxDefaultPosition, wxDefaultSize);
}

MainWindow::MainWindow(const wxString& title)
      : wxFrame(nullptr, wxID_ANY, title, wxDefaultPosition, wxSize(1900, 550)),
        picture(new Picture(new Picture::ColorFrame("photos/test_rgb.png"), nullptr, nullptr)),
        m_parent(new wxPanel(this, wxID_ANY)),
        m_display(new DisplayPanel(m_parent, picture)),
        m_settings(new SettingsPanel(m_parent, m_display)) {

   wxBoxSizer *hbox = new wxBoxSizer(wxHORIZONTAL);
   hbox->Add(m_settings, 1, wxEXPAND | wxALL, 5);

   m_parent->SetSizer(hbox);
   Centre();
}

// Main

class AppMain : public wxApp {
 public:
   virtual bool OnInit();
};

IMPLEMENT_APP(AppMain)

bool AppMain::OnInit() {
   MainWindow *window = new MainWindow(wxT("Simple display"));
   window->Show(true);

   return true;
}