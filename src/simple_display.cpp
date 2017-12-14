#include <wx/wx.h>
#include <wx/panel.h>
#include <wx/image.h>
#include <wx/slider.h>
#include <wx/wxprec.h>
#include <wx/bitmap.h>
#include <wx/stattext.h>

#include "rgbd_picture.h"

uint8_t constexpr device_type = 1;

enum {
    ID_MIN_D = 101,
    ID_MAX_D = 102
};

class display_panel;

class settings_panel : public wxPanel {
    display_panel *picture_panel;

    wxPanel *m_parent;
    wxSlider *m_min_d, *m_max_d;

public:
    settings_panel(wxPanel *parent, display_panel *picture_panel);

    void on_change(wxCommandEvent &event);
};

class display_panel : public wxPanel {
    rgbd_picture_t<device_type> *const picture;
    wxStaticBitmap *m_picture;

public:
    display_panel(wxPanel *parent, rgbd_picture_t<device_type> *picture);

    void update_picture(uint64_t const min_depth, uint64_t const max_depth);
};

class main_window : public wxFrame {
    rgbd_picture_t<device_type> *picture;
    wxPanel *m_parent;

    display_panel *m_display;
    settings_panel *m_settings;

public:
    main_window(const wxString& title);
};

// Definitions


settings_panel::settings_panel(wxPanel *parent, display_panel *picture_panel) : wxPanel(parent, -1,
  wxPoint(-1, -1), wxSize(-1, -1), wxBORDER_SUNKEN),
 picture_panel(picture_panel),
 m_parent(parent),
 m_min_d( new wxSlider(this, ID_MIN_D, 300, 0, 10000, wxPoint(10, 10), wxSize(980, 15)) ),
 m_max_d( new wxSlider(this, ID_MAX_D, 1000, 0, 10000, wxPoint(10, 40), wxSize(980, 15)) ) {
    Connect(ID_MIN_D, wxEVT_SCROLL_CHANGED, 
     wxCommandEventHandler(settings_panel::on_change));
    Connect(ID_MAX_D, wxEVT_SCROLL_CHANGED, 
     wxCommandEventHandler(settings_panel::on_change));
}

void settings_panel::on_change(wxCommandEvent &WXUNUSED(event)) {
  uint64_t const min_depth = m_min_d->GetValue();
  uint64_t const max_depth = m_max_d->GetValue();

  picture_panel->update_picture(min_depth, max_depth);
}

display_panel::display_panel(wxPanel *parent, rgbd_picture_t<device_type> *picture) : wxPanel(parent, wxID_ANY,
  wxDefaultPosition, wxSize(rgbd_picture_t<device_type>::width, rgbd_picture_t<device_type>::height),
  wxBORDER_SUNKEN), picture(picture) {
    wxImage tmp(picture->width, picture->height);
    tmp.SetData(picture->raw_bitmap());

    m_picture = new wxStaticBitmap(this, wxID_ANY, wxBitmap(tmp),
     wxDefaultPosition, wxDefaultSize);
}

void display_panel::update_picture(uint64_t const min_depth, uint64_t const max_depth) {
    picture->update_bitmap(min_depth, max_depth);

    wxImage tmp(picture->width, picture->height);
    tmp.SetData(picture->raw_bitmap());

    delete m_picture;
    m_picture = new wxStaticBitmap(this, wxID_ANY, wxBitmap(tmp),
     wxDefaultPosition, wxDefaultSize);
}

main_window::main_window(const wxString& title) : wxFrame(NULL, wxID_ANY, title,
  wxDefaultPosition, wxSize(1900, 550)),
 picture( new rgbd_picture_t<device_type>(std::string("kinect_test/photo_kinect1_depth.txt")) ),
 m_parent( new wxPanel(this, wxID_ANY) ),
 m_display( new display_panel(m_parent, picture) ),
 m_settings( new settings_panel(m_parent, m_display) ) {
    wxBoxSizer *hbox = new wxBoxSizer(wxHORIZONTAL);

    hbox->Add(m_settings, 1, wxEXPAND | wxALL, 5);
    hbox->Add(m_display, 1, wxEXPAND | wxALL, 5);

    m_parent->SetSizer(hbox);

    Centre();
}


// Main

class app_main : public wxApp {
  public:
    virtual bool OnInit();
};

IMPLEMENT_APP(app_main)

bool app_main::OnInit() {
    main_window *window = new main_window(wxT("Depth visible for eyes"));
    window->Show(true);

    return true;
}
